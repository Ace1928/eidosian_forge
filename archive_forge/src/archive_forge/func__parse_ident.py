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
@classmethod
def _parse_ident(cls, hash):
    """extract ident prefix from hash, helper for subclasses' from_string()"""
    hash = to_unicode(hash, 'ascii', 'hash')
    for ident in cls.ident_values:
        if hash.startswith(ident):
            return (ident, hash[len(ident):])
    raise exc.InvalidHashError(cls)