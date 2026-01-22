from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
@staticmethod
def _render_ini_value(key, value):
    """render value to string suitable for INI file"""
    if isinstance(value, (list, tuple)):
        value = ', '.join(value)
    elif isinstance(value, num_types):
        if isinstance(value, float) and key[2] == 'vary_rounds':
            value = ('%.2f' % value).rstrip('0') if value else '0'
        else:
            value = str(value)
    assert isinstance(value, native_string_types), 'expected string for key: %r %r' % (key, value)
    return value.replace('%', '%%')