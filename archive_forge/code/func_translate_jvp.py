from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def translate_jvp(jvpfun, fun, argnum):
    if jvpfun is None:
        return lambda g, ans, *a, **k: vspace(ans).zeros()
    elif jvpfun == 'same':
        return lambda g, ans, *args, **kwargs: fun(*subval(args, argnum, g), **kwargs)
    elif callable(jvpfun):
        return jvpfun
    else:
        raise Exception("Bad JVP '{}' for '{}'".format(jvpfun, fun.__name__))