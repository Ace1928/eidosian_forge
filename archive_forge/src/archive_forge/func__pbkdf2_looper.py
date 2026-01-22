from __future__ import division
import hashlib
import logging; log = logging.getLogger(__name__)
import re
import os
from struct import Struct
from warnings import warn
from passlib import exc
from passlib.utils import join_bytes, to_native_str, join_byte_values, to_bytes, \
from passlib.utils.compat import irange, int_types, unicode_or_bytes_types, PY3, error_from
from passlib.utils.decor import memoized_property
def _pbkdf2_looper(keyed_hmac, digest, rounds):
    hexlify = _hexlify
    accum = int(hexlify(digest), 16)
    for _ in irange(rounds - 1):
        digest = keyed_hmac(digest)
        accum ^= int(hexlify(digest), 16)
    return int_to_bytes(accum, len(digest))