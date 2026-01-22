from __future__ import (absolute_import, division, print_function)
import errno
import fcntl
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from binascii import hexlify
from binascii import unhexlify
from binascii import Error as BinasciiError
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible import constants as C
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe, unfrackpath
class VaultLib:

    def __init__(self, secrets=None):
        self.secrets = secrets or []
        self.cipher_name = None
        self.b_version = b'1.2'

    @staticmethod
    def is_encrypted(vaulttext):
        return is_encrypted(vaulttext)

    def encrypt(self, plaintext, secret=None, vault_id=None, salt=None):
        """Vault encrypt a piece of data.

        :arg plaintext: a text or byte string to encrypt.
        :returns: a utf-8 encoded byte str of encrypted data.  The string
            contains a header identifying this as vault encrypted data and
            formatted to newline terminated lines of 80 characters.  This is
            suitable for dumping as is to a vault file.

        If the string passed in is a text string, it will be encoded to UTF-8
        before encryption.
        """
        if secret is None:
            if self.secrets:
                dummy, secret = match_encrypt_secret(self.secrets)
            else:
                raise AnsibleVaultError('A vault password must be specified to encrypt data')
        b_plaintext = to_bytes(plaintext, errors='surrogate_or_strict')
        if is_encrypted(b_plaintext):
            raise AnsibleError('input is already encrypted')
        if not self.cipher_name or self.cipher_name not in CIPHER_WRITE_WHITELIST:
            self.cipher_name = u'AES256'
        try:
            this_cipher = CIPHER_MAPPING[self.cipher_name]()
        except KeyError:
            raise AnsibleError(u'{0} cipher could not be found'.format(self.cipher_name))
        if vault_id:
            display.vvvvv(u'Encrypting with vault_id "%s" and vault secret %s' % (to_text(vault_id), to_text(secret)))
        else:
            display.vvvvv(u'Encrypting without a vault_id using vault secret %s' % to_text(secret))
        b_ciphertext = this_cipher.encrypt(b_plaintext, secret, salt)
        b_vaulttext = format_vaulttext_envelope(b_ciphertext, self.cipher_name, vault_id=vault_id)
        return b_vaulttext

    def decrypt(self, vaulttext, filename=None, obj=None):
        """Decrypt a piece of vault encrypted data.

        :arg vaulttext: a string to decrypt.  Since vault encrypted data is an
            ascii text format this can be either a byte str or unicode string.
        :kwarg filename: a filename that the data came from.  This is only
            used to make better error messages in case the data cannot be
            decrypted.
        :returns: a byte string containing the decrypted data and the vault-id that was used

        """
        plaintext, vault_id, vault_secret = self.decrypt_and_get_vault_id(vaulttext, filename=filename, obj=obj)
        return plaintext

    def decrypt_and_get_vault_id(self, vaulttext, filename=None, obj=None):
        """Decrypt a piece of vault encrypted data.

        :arg vaulttext: a string to decrypt.  Since vault encrypted data is an
            ascii text format this can be either a byte str or unicode string.
        :kwarg filename: a filename that the data came from.  This is only
            used to make better error messages in case the data cannot be
            decrypted.
        :returns: a byte string containing the decrypted data and the vault-id vault-secret that was used

        """
        b_vaulttext = to_bytes(vaulttext, errors='strict', encoding='utf-8')
        if self.secrets is None:
            msg = 'A vault password must be specified to decrypt data'
            if filename:
                msg += ' in file %s' % to_native(filename)
            raise AnsibleVaultError(msg)
        if not is_encrypted(b_vaulttext):
            msg = 'input is not vault encrypted data. '
            if filename:
                msg += '%s is not a vault encrypted file' % to_native(filename)
            raise AnsibleError(msg)
        b_vaulttext, dummy, cipher_name, vault_id = parse_vaulttext_envelope(b_vaulttext, filename=filename)
        if cipher_name in CIPHER_WHITELIST:
            this_cipher = CIPHER_MAPPING[cipher_name]()
        else:
            raise AnsibleError('{0} cipher could not be found'.format(cipher_name))
        b_plaintext = None
        if not self.secrets:
            raise AnsibleVaultError('Attempting to decrypt but no vault secrets found')
        vault_id_matchers = []
        vault_id_used = None
        vault_secret_used = None
        if vault_id:
            display.vvvvv(u'Found a vault_id (%s) in the vaulttext' % to_text(vault_id))
            vault_id_matchers.append(vault_id)
            _matches = match_secrets(self.secrets, vault_id_matchers)
            if _matches:
                display.vvvvv(u'We have a secret associated with vault id (%s), will try to use to decrypt %s' % (to_text(vault_id), to_text(filename)))
            else:
                display.vvvvv(u'Found a vault_id (%s) in the vault text, but we do not have a associated secret (--vault-id)' % to_text(vault_id))
        if not C.DEFAULT_VAULT_ID_MATCH:
            vault_id_matchers.extend([_vault_id for _vault_id, _dummy in self.secrets if _vault_id != vault_id])
        matched_secrets = match_secrets(self.secrets, vault_id_matchers)
        for vault_secret_id, vault_secret in matched_secrets:
            display.vvvvv(u'Trying to use vault secret=(%s) id=%s to decrypt %s' % (to_text(vault_secret), to_text(vault_secret_id), to_text(filename)))
            try:
                display.vvvv(u'Trying secret %s for vault_id=%s' % (to_text(vault_secret), to_text(vault_secret_id)))
                b_plaintext = this_cipher.decrypt(b_vaulttext, vault_secret)
                if b_plaintext is not None:
                    vault_id_used = vault_secret_id
                    vault_secret_used = vault_secret
                    file_slug = ''
                    if filename:
                        file_slug = ' of "%s"' % filename
                    display.vvvvv(u'Decrypt%s successful with secret=%s and vault_id=%s' % (to_text(file_slug), to_text(vault_secret), to_text(vault_secret_id)))
                    break
            except AnsibleVaultFormatError as exc:
                exc.obj = obj
                msg = u'There was a vault format error'
                if filename:
                    msg += u' in %s' % to_text(filename)
                msg += u': %s' % to_text(exc)
                display.warning(msg, formatted=True)
                raise
            except AnsibleError as e:
                display.vvvv(u'Tried to use the vault secret (%s) to decrypt (%s) but it failed. Error: %s' % (to_text(vault_secret_id), to_text(filename), e))
                continue
        else:
            msg = 'Decryption failed (no vault secrets were found that could decrypt)'
            if filename:
                msg += ' on %s' % to_native(filename)
            raise AnsibleVaultError(msg)
        if b_plaintext is None:
            msg = 'Decryption failed'
            if filename:
                msg += ' on %s' % to_native(filename)
            raise AnsibleError(msg)
        return (b_plaintext, vault_id_used, vault_secret_used)