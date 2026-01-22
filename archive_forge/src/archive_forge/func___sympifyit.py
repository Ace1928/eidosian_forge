from functools import wraps
from .sympify import SympifyError, sympify
def __sympifyit(func, arg, retval=None):
    """Decorator to _sympify `arg` argument for function `func`.

       Do not use directly -- use _sympifyit instead.
    """
    if not func.__code__.co_argcount:
        raise LookupError('func not found')
    assert func.__code__.co_varnames[1] == arg
    if retval is None:

        @wraps(func)
        def __sympifyit_wrapper(a, b):
            return func(a, sympify(b, strict=True))
    else:

        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                if not hasattr(b, '_op_priority'):
                    b = sympify(b, strict=True)
                return func(a, b)
            except SympifyError:
                return retval
    return __sympifyit_wrapper