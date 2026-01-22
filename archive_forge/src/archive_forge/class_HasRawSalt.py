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
class HasRawSalt(HasSalt):
    """mixin for classes which use decoded salt parameter

    A variant of :class:`!HasSalt` which takes in decoded bytes instead of an encoded string.

    .. todo::

        document this class's usage
    """
    salt_chars = ALL_BYTE_VALUES
    _salt_is_bytes = True
    _salt_unit = 'bytes'

    @classmethod
    def _generate_salt(cls):
        assert cls.salt_chars in [None, ALL_BYTE_VALUES]
        return getrandbytes(rng, cls.default_salt_size)