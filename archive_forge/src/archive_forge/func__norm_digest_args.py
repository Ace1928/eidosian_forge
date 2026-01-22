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
@classmethod
def _norm_digest_args(cls, secret, ident, new=False):
    require_valid_utf8_bytes = cls._require_valid_utf8_bytes
    if isinstance(secret, unicode):
        secret = secret.encode('utf-8')
    elif require_valid_utf8_bytes:
        try:
            secret.decode('utf-8')
        except UnicodeDecodeError:
            require_valid_utf8_bytes = False
    uh.validate_secret(secret)
    if new:
        cls._check_truncate_policy(secret)
    if _BNULL in secret:
        raise uh.exc.NullPasswordError(cls)
    if cls._has_2a_wraparound_bug and len(secret) >= 255:
        if require_valid_utf8_bytes:
            secret = utf8_truncate(secret, 72)
        else:
            secret = secret[:72]
    if ident == IDENT_2A:
        pass
    elif ident == IDENT_2B:
        if cls._lacks_2b_support:
            ident = cls._fallback_ident
    elif ident == IDENT_2Y:
        if cls._lacks_2y_support:
            ident = cls._fallback_ident
    elif ident == IDENT_2:
        if cls._lacks_20_support:
            if secret:
                if require_valid_utf8_bytes:
                    secret = utf8_repeat_string(secret, 72)
                else:
                    secret = repeat_string(secret, 72)
            ident = cls._fallback_ident
    elif ident == IDENT_2X:
        raise RuntimeError('$2x$ hashes not currently supported by passlib')
    else:
        raise AssertionError('unexpected ident value: %r' % ident)
    return (secret, ident)