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
class _wrapped_bcrypt(bcrypt):
    """
    abstracts out some bits bcrypt_sha256 & django_bcrypt_sha256 share.
    - bypass backend-loading wrappers for hash() etc
    - disable truncation support, sha256 wrappers don't need it.
    """
    setting_kwds = tuple((elem for elem in bcrypt.setting_kwds if elem not in ['truncate_error']))
    truncate_size = None

    @classmethod
    def _check_truncate_policy(cls, secret):
        pass