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
def _stub_requires_backend(cls):
    """
        helper for subclasses to create stub methods which auto-load backend.
        """
    if cls.__backend:
        raise AssertionError('%s: _finalize_backend(%r) failed to replace lazy loader' % (cls.name, cls.__backend))
    cls.set_backend()
    if not cls.__backend:
        raise AssertionError('%s: set_backend() failed to load a default backend' % cls.name)