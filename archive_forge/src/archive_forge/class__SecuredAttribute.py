from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
class _SecuredAttribute(object):

    def __init__(self, obj, check_permissions):
        self.obj = obj
        self.check_permissions = check_permissions
        self._parent = None

    def _check_permissions(self):
        if isinstance(self.check_permissions, str):
            return getattr(self.parent, self.check_permissions)()
        else:
            return self.check_permissions()

    def __get_parent(self):
        return self._parent

    def __set_parent(self, parent):
        if ismethod(parent):
            self._parent = parent.__self__
        else:
            self._parent = parent
    parent = property(__get_parent, __set_parent)

    @_secure_method('_check_permissions')
    @expose()
    def _lookup(self, *remainder):
        return (self.obj, list(remainder))