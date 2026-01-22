import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
class _PasslibRegistryProxy(object):
    """proxy module passlib.hash

    this module is in fact an object which lazy-loads
    the requested password hash algorithm from wherever it has been stored.
    it acts as a thin wrapper around :func:`passlib.registry.get_crypt_handler`.
    """
    __name__ = 'passlib.hash'
    __package__ = None

    def __getattr__(self, attr):
        if attr.startswith('_'):
            raise AttributeError('missing attribute: %r' % (attr,))
        handler = get_crypt_handler(attr, None)
        if handler:
            return handler
        else:
            raise AttributeError('unknown password hash: %r' % (attr,))

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        else:
            register_crypt_handler(value, _attr=attr)

    def __repr__(self):
        return "<proxy module 'passlib.hash'>"

    def __dir__(self):
        attrs = set(dir(self.__class__))
        attrs.update(self.__dict__)
        attrs.update(_locations)
        return sorted(attrs)