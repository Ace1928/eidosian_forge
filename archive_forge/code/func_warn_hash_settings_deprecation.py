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
def warn_hash_settings_deprecation(handler, kwds):
    warn("passing settings to %(handler)s.hash() is deprecated, and won't be supported in Passlib 2.0; use '%(handler)s.using(**settings).hash(secret)' instead" % dict(handler=handler.name), DeprecationWarning, stacklevel=guess_app_stacklevel(2))