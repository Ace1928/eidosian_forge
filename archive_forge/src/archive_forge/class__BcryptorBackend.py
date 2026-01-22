from __future__ import with_statement, absolute_import
from base64 import b64encode
from hashlib import sha256
import os
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.crypto.digest import compile_hmac
from passlib.exc import PasslibHashWarning, PasslibSecurityWarning, PasslibSecurityError
from passlib.utils import safe_crypt, repeat_string, to_bytes, parse_version, \
from passlib.utils.binary import bcrypt64
from passlib.utils.compat import get_unbound_method_function
from passlib.utils.compat import u, uascii_to_str, unicode, str_to_uascii, PY3, error_from
import passlib.utils.handlers as uh
class _BcryptorBackend(_BcryptCommon):
    """
    backend which uses 'bcryptor' package
    """

    @classmethod
    def _load_backend_mixin(mixin_cls, name, dryrun):
        global _bcryptor
        try:
            import bcryptor as _bcryptor
        except ImportError:
            return False
        if not dryrun:
            warn('Support for `bcryptor` is deprecated, and will be removed in Passlib 1.8; Please use `pip install bcrypt` instead', DeprecationWarning)
        return mixin_cls._finalize_backend_mixin(name, dryrun)

    def _calc_checksum(self, secret):
        secret, ident = self._prepare_digest_args(secret)
        config = self._get_config(ident)
        hash = _bcryptor.engine.Engine(False).hash_key(secret, config)
        if not hash.startswith(config) or len(hash) != len(config) + 31:
            raise uh.exc.CryptBackendError(self, config, hash, source='bcryptor library')
        return str_to_uascii(hash[-31:])