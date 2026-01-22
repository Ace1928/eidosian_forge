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
@classmethod
def _adapt_dict_kwds(cls, type, **kwds):
    """
        Internal helper for .from_json() --
        Adapts serialized json dict into constructor keywords.
        """
    assert cls._check_otp_type(type)
    ver = kwds.pop('v', None)
    if not ver or ver < cls.min_json_version or ver > cls.json_version:
        raise cls._dict_parse_error('missing/unsupported version (%r)' % (ver,))
    elif ver != cls.json_version:
        kwds['changed'] = True
    if 'enckey' in kwds:
        assert 'key' not in kwds
        kwds.update(key=kwds.pop('enckey'), format='encrypted')
    elif 'key' not in kwds:
        raise cls._dict_parse_error("missing 'enckey' / 'key'")
    kwds.pop('last_counter', None)
    return kwds