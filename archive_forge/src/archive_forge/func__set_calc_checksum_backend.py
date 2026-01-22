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
def _set_calc_checksum_backend(cls, func):
    """
        helper used by subclasses to validate & set backend-specific
        calc checksum helper.
        """
    backend = cls._pending_backend
    assert backend, 'should only be called during set_backend()'
    if not callable(func):
        raise RuntimeError('%s: backend %r returned invalid callable: %r' % (cls.name, backend, func))
    if not cls._pending_dry_run:
        cls._calc_checksum_backend = func