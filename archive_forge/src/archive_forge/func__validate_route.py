import logging
import re
import warnings
from inspect import getmembers, ismethod
from webob import exc
from .secure import handle_security, cross_boundary
from .util import iscontroller, getargspec, _cfg
def _validate_route(route):
    if not isinstance(route, str):
        raise TypeError('%s must be a string' % route)
    if route in ('.', '..') or not re.match('^[0-9a-zA-Z-_$\\(\\)\\.~!,;:*+@=]+$', route):
        raise ValueError('%s must be a valid path segment.  Keep in mind that path segments should not contain path separators (e.g., /) ' % route)