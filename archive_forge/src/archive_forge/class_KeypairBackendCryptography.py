from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
class KeypairBackendCryptography(KeypairBackend):

    def __init__(self, module):
        super(KeypairBackendCryptography, self).__init__(module)
        if self.type == 'rsa1':
            self.module.fail_json(msg='RSA1 keys are not supported by the cryptography backend')
        self.passphrase = to_bytes(module.params['passphrase']) if module.params['passphrase'] else None
        self.private_key_format = self._get_key_format(module.params['private_key_format'])

    def _get_key_format(self, key_format):
        result = 'SSH'
        if key_format == 'auto':
            ssh_version = self._get_ssh_version() or '7.8'
            if LooseVersion(ssh_version) < LooseVersion('7.8') and self.type != 'ed25519':
                result = 'PKCS1'
            if result == 'SSH' and (not HAS_OPENSSH_PRIVATE_FORMAT):
                self.module.fail_json(msg=missing_required_lib('cryptography >= 3.0', reason='to load/dump private keys in the default OpenSSH format for OpenSSH >= 7.8 ' + 'or for ed25519 keys'))
        else:
            result = key_format.upper()
        return result

    def _generate_keypair(self, private_key_path):
        keypair = OpensshKeypair.generate(keytype=self.type, size=self.size, passphrase=self.passphrase, comment=self.comment or '')
        encoded_private_key = OpensshKeypair.encode_openssh_privatekey(keypair.asymmetric_keypair, self.private_key_format)
        secure_write(private_key_path, 384, encoded_private_key)
        public_key_path = private_key_path + '.pub'
        secure_write(public_key_path, 420, keypair.public_key)

    def _get_private_key(self):
        keypair = OpensshKeypair.load(path=self.private_key_path, passphrase=self.passphrase, no_public_key=True)
        return PrivateKey(size=keypair.size, key_type=keypair.key_type, fingerprint=keypair.fingerprint, format=parse_private_key_format(self.private_key_path))

    def _get_public_key(self):
        try:
            keypair = OpensshKeypair.load(path=self.private_key_path, passphrase=self.passphrase, no_public_key=True)
        except OpenSSHError:
            return ''
        return PublicKey.from_string(to_text(keypair.public_key))

    def _private_key_readable(self):
        try:
            OpensshKeypair.load(path=self.private_key_path, passphrase=self.passphrase, no_public_key=True)
        except (InvalidPrivateKeyFileError, InvalidPassphraseError):
            return False
        if self.passphrase:
            try:
                OpensshKeypair.load(path=self.private_key_path, passphrase=None, no_public_key=True)
            except (InvalidPrivateKeyFileError, InvalidPassphraseError):
                return True
            else:
                return False
        return True

    def _update_comment(self):
        keypair = OpensshKeypair.load(path=self.private_key_path, passphrase=self.passphrase, no_public_key=True)
        try:
            keypair.comment = self.comment
        except InvalidCommentError as e:
            self.module.fail_json(msg=to_native(e))
        try:
            temp_public_key = self._create_temp_public_key(keypair.public_key + b'\n')
            self._safe_secure_move([(temp_public_key, self.public_key_path)])
        except (IOError, OSError) as e:
            self.module.fail_json(msg=to_native(e))

    def _private_key_valid_backend(self):
        if self.module.params['private_key_format'] == 'auto':
            return True
        return self.private_key_format == self.original_private_key.format