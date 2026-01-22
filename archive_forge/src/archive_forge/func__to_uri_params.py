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
def _to_uri_params(self):
    """return list of (key, param) entries for URI"""
    args = [('secret', self.base32_key)]
    if self.alg != 'sha1':
        args.append(('algorithm', self.alg.upper()))
    if self.digits != 6:
        args.append(('digits', str(self.digits)))
    if self.period != 30:
        args.append(('period', str(self.period)))
    return args