from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def vjp_argnums(argnums, ans, args, kwargs):
    L = len(argnums)
    if L == 1:
        argnum = argnums[0]
        try:
            vjpfun = vjps_dict[argnum]
        except KeyError:
            raise NotImplementedError('VJP of {} wrt argnum 0 not defined'.format(fun.__name__))
        vjp = vjpfun(ans, *args, **kwargs)
        return lambda g: (vjp(g),)
    elif L == 2:
        argnum_0, argnum_1 = argnums
        try:
            vjp_0_fun = vjps_dict[argnum_0]
            vjp_1_fun = vjps_dict[argnum_1]
        except KeyError:
            raise NotImplementedError('VJP of {} wrt argnums 0, 1 not defined'.format(fun.__name__))
        vjp_0 = vjp_0_fun(ans, *args, **kwargs)
        vjp_1 = vjp_1_fun(ans, *args, **kwargs)
        return lambda g: (vjp_0(g), vjp_1(g))
    else:
        vjps = [vjps_dict[argnum](ans, *args, **kwargs) for argnum in argnums]
        return lambda g: (vjp(g) for vjp in vjps)