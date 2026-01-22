import functools
import inspect
import sys
from pyomo.common import DeveloperError
def _disable_property(fcn, msg=None, exception=RuntimeError):
    _name = fcn.fget.__name__
    if msg is None:
        _gmsg = "access property '%s' on" % (_name,)
    else:
        _gmsg = msg
    getter = _disable_method(fcn.fget, _gmsg, exception)
    if fcn.fset is None:
        setter = None
    else:
        if msg is None:
            _smsg = "set property '%s' on" % (_name,)
        else:
            _smsg = msg
        setter = _disable_method(fcn.fset, _smsg, exception)
    return property(fget=getter, fset=setter, doc=fcn.__doc__)