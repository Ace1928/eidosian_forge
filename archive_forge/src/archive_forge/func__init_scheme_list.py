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
def _init_scheme_list(self, data):
    """initialize .handlers and .schemes attributes"""
    handlers = []
    schemes = []
    if isinstance(data, native_string_types):
        data = splitcomma(data)
    for elem in data or ():
        if hasattr(elem, 'name'):
            handler = elem
            scheme = handler.name
            _validate_handler_name(scheme)
        elif isinstance(elem, native_string_types):
            handler = get_crypt_handler(elem)
            scheme = handler.name
        else:
            raise TypeError('scheme must be name or CryptHandler, not %r' % type(elem))
        if scheme in schemes:
            raise KeyError('multiple handlers with same name: %r' % (scheme,))
        handlers.append(handler)
        schemes.append(scheme)
    self.handlers = tuple(handlers)
    self.schemes = tuple(schemes)