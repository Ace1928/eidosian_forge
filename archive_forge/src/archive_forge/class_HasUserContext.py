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
class HasUserContext(GenericHandler):
    """helper for classes which require a user context keyword"""
    context_kwds = ('user',)

    def __init__(self, user=None, **kwds):
        super(HasUserContext, self).__init__(**kwds)
        self.user = user

    @classmethod
    def hash(cls, secret, user=None, **context):
        return super(HasUserContext, cls).hash(secret, user=user, **context)

    @classmethod
    def verify(cls, secret, hash, user=None, **context):
        return super(HasUserContext, cls).verify(secret, hash, user=user, **context)

    @deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genhash(cls, secret, config, user=None, **context):
        return super(HasUserContext, cls).genhash(secret, config, user=user, **context)