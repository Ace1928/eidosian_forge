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
class MinimalHandler(PasswordHash):
    """
    helper class for implementing hash handlers.
    provides nothing besides a base implementation of the .using() subclass constructor.
    """
    _configured = False

    @classmethod
    def using(cls, relaxed=False):
        name = cls.__name__
        if not cls._configured:
            name = '<customized %s hasher>' % name
        return type(name, (cls,), dict(__module__=cls.__module__, _configured=True))