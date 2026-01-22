from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
def _wrap_hash(self, hash):
    """given orig hash; return one belonging to wrapper"""
    if isinstance(hash, bytes):
        hash = hash.decode('ascii')
    orig_prefix = self.orig_prefix
    if not hash.startswith(orig_prefix):
        raise exc.InvalidHashError(self.wrapped)
    wrapped = self.prefix + hash[len(orig_prefix):]
    return uascii_to_str(wrapped)