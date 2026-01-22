from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def _update_function(self, oldfunc, newfunc):
    """Update a function object."""
    oldfunc.__doc__ = newfunc.__doc__
    oldfunc.__dict__.update(newfunc.__dict__)
    try:
        newfunc.__code__
        attr_name = '__code__'
    except AttributeError:
        newfunc.func_code
        attr_name = 'func_code'
    old_code = getattr(oldfunc, attr_name)
    new_code = getattr(newfunc, attr_name)
    if not code_objects_equal(old_code, new_code):
        notify_info0('Updated function code:', oldfunc)
        setattr(oldfunc, attr_name, new_code)
        self.found_change = True
    try:
        oldfunc.__defaults__ = newfunc.__defaults__
    except AttributeError:
        oldfunc.func_defaults = newfunc.func_defaults
    return oldfunc