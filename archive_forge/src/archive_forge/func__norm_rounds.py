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
def _norm_rounds(cls, rounds, relaxed=False, param='rounds'):
    """
        helper for normalizing rounds value.

        :arg rounds:
            an integer cost parameter.

        :param relaxed:
            if ``True`` (the default), issues PasslibHashWarning is rounds are outside allowed range.
            if ``False``, raises a ValueError instead.

        :param param:
            optional name of parameter to insert into error/warning messages.

        :raises TypeError:
            * if ``use_defaults=False`` and no rounds is specified
            * if rounds is not an integer.

        :raises ValueError:

            * if rounds is ``None`` and class does not specify a value for
              :attr:`default_rounds`.
            * if ``relaxed=False`` and rounds is outside bounds of
              :attr:`min_rounds` and :attr:`max_rounds` (if ``relaxed=True``,
              the rounds value will be clamped, and a warning issued).

        :returns:
            normalized rounds value
        """
    return norm_integer(cls, rounds, cls.min_rounds, cls.max_rounds, param=param, relaxed=relaxed)