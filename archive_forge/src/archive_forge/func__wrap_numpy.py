from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def _wrap_numpy(k):
    numpy_func = getattr(np, k)
    if sys.version_info[0] > 2:
        from functools import wraps
    else:

        def wraps(_meta_fun):
            return lambda x: x

    @wraps(numpy_func)
    def f(*args, **kwargs):
        return numpy_func(*map(to_unitless, args), **kwargs)
    return f