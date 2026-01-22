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
def __load_legacy_backend(cls, name):
    value = getattr(cls, '_has_backend_' + name)
    warn('%s: support for ._has_backend_%s is deprecated as of Passlib 1.7, and will be removed in Passlib 1.9/2.0, please implement ._load_backend_%s() instead' % (cls.name, name, name), DeprecationWarning)
    if value:
        func = getattr(cls, '_calc_checksum_' + name)
        cls._set_calc_checksum_backend(func)
        return True
    else:
        return False