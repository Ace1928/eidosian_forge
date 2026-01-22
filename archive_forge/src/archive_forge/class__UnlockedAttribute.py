from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
class _UnlockedAttribute(object):

    def __init__(self, obj):
        self.obj = obj

    @_unlocked_method
    @expose()
    def _lookup(self, *remainder):
        return (self.obj, remainder)