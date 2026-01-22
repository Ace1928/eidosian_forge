from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
class SecureControllerMeta(type):
    """
    Used to apply security to a controller.
    Implementations of SecureController should extend the
    `check_permissions` method to return a True or False
    value (depending on whether or not the user has permissions
    to the controller).
    """

    def __init__(cls, name, bases, dict_):
        cls._pecan = dict(secured=Protected, check_permissions=cls.check_permissions, unlocked=[])
        for name, value in getmembers(cls)[:]:
            if isfunction(value):
                if iscontroller(value) and value._pecan.get('secured') is None:
                    wrapped = _make_wrapper(value)
                    wrapped._pecan['secured'] = Protected
                    wrapped._pecan['check_permissions'] = cls.check_permissions
                    setattr(cls, name, wrapped)
            elif hasattr(value, '__class__'):
                if name.startswith('__') and name.endswith('__'):
                    continue
                if isinstance(value, _UnlockedAttribute):
                    cls._pecan['unlocked'].append(value.obj)
                    setattr(cls, name, value.obj)
                elif isinstance(value, _SecuredAttribute):
                    cls._pecan['unlocked'].append(value)