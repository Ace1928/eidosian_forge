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
def _norm_checksum(self, checksum, relaxed=False):
    checksum = super(_BcryptCommon, self)._norm_checksum(checksum, relaxed=relaxed)
    changed, checksum = bcrypt64.check_repair_unused(checksum)
    if changed:
        warn('encountered a bcrypt hash with incorrectly set padding bits; you may want to use bcrypt.normhash() to fix this; this will be an error under Passlib 2.0', PasslibHashWarning)
    return checksum