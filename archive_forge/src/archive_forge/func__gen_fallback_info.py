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
def _gen_fallback_info():
    """
    internal helper used to generate ``_fallback_info`` dict.
    currently only run manually to update the above list;
    not invoked at runtime.
    """
    out = {}
    for alg in sorted(hashlib.algorithms_available | set(['md4'])):
        info = lookup_hash(alg)
        out[info.name] = (info.digest_size, info.block_size)
    return out