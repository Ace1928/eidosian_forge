import inspect
from functools import partial
from weakref import WeakMethod
@staticmethod
def _raise_dispatch_exception(event_type, args, handler, exception):
    n_args = len(args)
    argspecs = inspect.getfullargspec(handler)
    handler_args = argspecs.args
    handler_varargs = argspecs.varargs
    handler_defaults = argspecs.defaults
    n_handler_args = len(handler_args)
    if inspect.ismethod(handler) and handler.__self__:
        n_handler_args -= 1
    if handler_varargs:
        n_handler_args = max(n_handler_args, n_args)
    if n_handler_args > n_args >= n_handler_args - len(handler_defaults) and handler_defaults:
        n_handler_args = n_args
    if n_handler_args != n_args:
        if inspect.isfunction(handler) or inspect.ismethod(handler):
            descr = f"'{handler.__name__}' at {handler.__code__.co_filename}:{handler.__code__.co_firstlineno}"
        else:
            descr = repr(handler)
        raise TypeError(f"The '{event_type}' event was dispatched with {len(args)} arguments,\nbut your handler {descr} accepts only {n_handler_args} arguments.")
    else:
        raise exception