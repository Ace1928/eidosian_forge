from __future__ import absolute_import, division, print_function
from passlib.utils.compat import PY3
import base64
import calendar
import json
import logging; log = logging.getLogger(__name__)
import math
import struct
import sys
import time as _time
import re
from warnings import warn
from passlib import exc
from passlib.exc import TokenError, MalformedTokenError, InvalidTokenError, UsedTokenError
from passlib.utils import (to_unicode, to_bytes, consteq,
from passlib.utils.binary import BASE64_CHARS, b32encode, b32decode
from passlib.utils.compat import (u, unicode, native_string_types, bascii_to_str, int_types, num_types,
from passlib.utils.decor import hybrid_method, memoized_property
from passlib.crypto.digest import lookup_hash, compile_hmac, pbkdf2_hmac
from passlib.hash import pbkdf2_sha256
class AppWallet(object):
    """
    This class stores application-wide secrets that can be used
    to encrypt & decrypt TOTP keys for storage.
    It's mostly an internal detail, applications usually just need
    to pass ``secrets`` or ``secrets_path`` to :meth:`TOTP.using`.

    .. seealso::

        :ref:`totp-storing-instances` for more details on this workflow.

    Arguments
    =========
    :param secrets:
        Dict of application secrets to use when encrypting/decrypting
        stored TOTP keys.  This should include a secret to use when encrypting
        new keys, but may contain additional older secrets to decrypt
        existing stored keys.

        The dict should map tags -> secrets, so that each secret is identified
        by a unique tag.  This tag will be stored along with the encrypted
        key in order to determine which secret should be used for decryption.
        Tag should be string that starts with regex range ``[a-z0-9]``,
        and the remaining characters must be in ``[a-z0-9_.-]``.

        It is recommended to use something like a incremental counter
        ("1", "2", ...), an ISO date ("2016-01-01", "2016-05-16", ...), 
        or a timestamp ("19803495", "19813495", ...) when assigning tags.

        This mapping be provided in three formats:

        * A python dict mapping tag -> secret
        * A JSON-formatted string containing the dict
        * A multiline string with the format ``"tag: value\\ntag: value\\n..."``

        (This last format is mainly useful when loading from a text file via **secrets_path**)

        .. seealso:: :func:`generate_secret` to create a secret with sufficient entropy

    :param secrets_path:
        Alternately, callers can specify a separate file where the
        application-wide secrets are stored, using either of the string
        formats described in **secrets**.

    :param default_tag:
        Specifies which tag in **secrets** should be used as the default
        for encrypting new keys. If omitted, the tags will be sorted,
        and the largest tag used as the default.

        if all tags are numeric, they will be sorted numerically;
        otherwise they will be sorted alphabetically.
        this permits tags to be assigned numerically,
        or e.g. using ``YYYY-MM-DD`` dates.

    :param encrypt_cost:
        Optional time-cost factor for key encryption.
        This value corresponds to log2() of the number of PBKDF2
        rounds used.

    .. warning::

        The application secret(s) should be stored in a secure location by
        your application, and each secret should contain a large amount
        of entropy (to prevent brute-force attacks if the encrypted keys
        are leaked).

        :func:`generate_secret` is provided as a convenience helper
        to generate a new application secret of suitable size.

        Best practice is to load these values from a file via **secrets_path**,
        and then have your application give up permission to read this file
        once it's running.

    Public Methods
    ==============
    .. autoattribute:: has_secrets
    .. autoattribute:: default_tag

    Semi-Private Methods
    ====================
    The following methods are used internally by the :class:`TOTP`
    class in order to encrypt & decrypt keys using the provided application
    secrets.  They will generally not be publically useful, and may have their
    API changed periodically.

    .. automethod:: get_secret
    .. automethod:: encrypt_key
    .. automethod:: decrypt_key
    """
    salt_size = 12
    encrypt_cost = 14
    _secrets = None
    default_tag = None

    def __init__(self, secrets=None, default_tag=None, encrypt_cost=None, secrets_path=None):
        if encrypt_cost is not None:
            if isinstance(encrypt_cost, native_string_types):
                encrypt_cost = int(encrypt_cost)
            assert encrypt_cost >= 0
            self.encrypt_cost = encrypt_cost
        if secrets_path is not None:
            if secrets is not None:
                raise TypeError("'secrets' and 'secrets_path' are mutually exclusive")
            secrets = open(secrets_path, 'rt').read()
        secrets = self._secrets = self._parse_secrets(secrets)
        if secrets:
            if default_tag is not None:
                self.get_secret(default_tag)
            elif all((tag.isdigit() for tag in secrets)):
                default_tag = max(secrets, key=int)
            else:
                default_tag = max(secrets)
            self.default_tag = default_tag

    def _parse_secrets(self, source):
        """
        parse 'secrets' parameter

        :returns:
            Dict[tag:str, secret:bytes]
        """
        check_type = True
        if isinstance(source, native_string_types):
            if source.lstrip().startswith(('[', '{')):
                source = json.loads(source)
            elif '\n' in source and ':' in source:

                def iter_pairs(source):
                    for line in source.splitlines():
                        line = line.strip()
                        if line and (not line.startswith('#')):
                            tag, secret = line.split(':', 1)
                            yield (tag.strip(), secret.strip())
                source = iter_pairs(source)
                check_type = False
            else:
                raise ValueError('unrecognized secrets string format')
        if source is None:
            return {}
        elif isinstance(source, dict):
            source = source.items()
        elif check_type:
            raise TypeError("'secrets' must be mapping, or list of items")
        return dict((self._parse_secret_pair(tag, value) for tag, value in source))

    def _parse_secret_pair(self, tag, value):
        if isinstance(tag, native_string_types):
            pass
        elif isinstance(tag, int):
            tag = str(tag)
        else:
            raise TypeError('tag must be unicode/string: %r' % (tag,))
        if not _tag_re.match(tag):
            raise ValueError('tag contains invalid characters: %r' % (tag,))
        if not isinstance(value, bytes):
            value = to_bytes(value, param='secret %r' % (tag,))
        if not value:
            raise ValueError('tag contains empty secret: %r' % (tag,))
        return (tag, value)

    @property
    def has_secrets(self):
        """whether at least one application secret is present"""
        return self.default_tag is not None

    def get_secret(self, tag):
        """
        resolve a secret tag to the secret (as bytes).
        throws a KeyError if not found.
        """
        secrets = self._secrets
        if not secrets:
            raise KeyError('no application secrets configured')
        try:
            return secrets[tag]
        except KeyError:
            raise suppress_cause(KeyError('unknown secret tag: %r' % (tag,)))

    @staticmethod
    def _cipher_aes_key(value, secret, salt, cost, decrypt=False):
        """
        Internal helper for :meth:`encrypt_key` --
        handles lowlevel encryption/decryption.

        Algorithm details:

        This function uses PBKDF2-HMAC-SHA256 to generate a 32-byte AES key
        and a 16-byte IV from the application secret & random salt.
        It then uses AES-256-CTR to encrypt/decrypt the TOTP key.

        CTR mode was chosen over CBC because the main attack scenario here
        is that the attacker has stolen the database, and is trying to decrypt a TOTP key
        (the plaintext value here).  To make it hard for them, we want every password
        to decrypt to a potentially valid key -- thus need to avoid any authentication
        or padding oracle attacks.  While some random padding construction could be devised
        to make this work for CBC mode, a stream cipher mode is just plain simpler.
        OFB/CFB modes would also work here, but seeing as they have malleability
        and cyclic issues (though remote and barely relevant here),
        CTR was picked as the best overall choice.
        """
        if _cg_ciphers is None:
            raise RuntimeError("TOTP encryption requires 'cryptography' package (https://cryptography.io)")
        keyiv = pbkdf2_hmac('sha256', secret, salt=salt, rounds=1 << cost, keylen=48)
        cipher = _cg_ciphers.Cipher(_cg_ciphers.algorithms.AES(keyiv[:32]), _cg_ciphers.modes.CTR(keyiv[32:]), _cg_default_backend())
        ctx = cipher.decryptor() if decrypt else cipher.encryptor()
        return ctx.update(value) + ctx.finalize()

    def encrypt_key(self, key):
        """
        Helper used to encrypt TOTP keys for storage.

        :param key:
            TOTP key to encrypt, as raw bytes.

        :returns:
            dict containing encrypted TOTP key & configuration parameters.
            this format should be treated as opaque, and potentially subject
            to change, though it is designed to be easily serialized/deserialized
            (e.g. via JSON).

        .. note::

            This function requires installation of the external
            `cryptography <https://cryptography.io>`_ package.

        To give some algorithm details:  This function uses AES-256-CTR to encrypt
        the provided data.  It takes the application secret and randomly generated salt,
        and uses PBKDF2-HMAC-SHA256 to combine them and generate the AES key & IV.
        """
        if not key:
            raise ValueError('no key provided')
        salt = getrandbytes(rng, self.salt_size)
        cost = self.encrypt_cost
        tag = self.default_tag
        if not tag:
            raise TypeError("no application secrets configured, can't encrypt OTP key")
        ckey = self._cipher_aes_key(key, self.get_secret(tag), salt, cost)
        return dict(v=1, c=cost, t=tag, s=b32encode(salt), k=b32encode(ckey))

    def decrypt_key(self, enckey):
        """
        Helper used to decrypt TOTP keys from storage format.
        Consults configured secrets to decrypt key.

        :param source:
            source object, as returned by :meth:`encrypt_key`.

        :returns:
            ``(key, needs_recrypt)`` --

            **key** will be the decrypted key, as bytes.

            **needs_recrypt** will be a boolean flag indicating
            whether encryption cost or default tag is too old,
            and henace that key needs re-encrypting before storing.

        .. note::

            This function requires installation of the external
            `cryptography <https://cryptography.io>`_ package.
        """
        if not isinstance(enckey, dict):
            raise TypeError("'enckey' must be dictionary")
        version = enckey.get('v', None)
        needs_recrypt = False
        if version == 1:
            _cipher_key = self._cipher_aes_key
        else:
            raise ValueError("missing / unrecognized 'enckey' version: %r" % (version,))
        tag = enckey['t']
        cost = enckey['c']
        key = _cipher_key(value=b32decode(enckey['k']), secret=self.get_secret(tag), salt=b32decode(enckey['s']), cost=cost)
        if cost != self.encrypt_cost or tag != self.default_tag:
            needs_recrypt = True
        return (key, needs_recrypt)