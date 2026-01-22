from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
class quirks:
    none_causes_check_password_error = DJANGO_VERSION >= (2, 1)
    empty_is_usable_password = DJANGO_VERSION >= (2, 1)
    invalid_is_usable_password = DJANGO_VERSION >= (2, 1)