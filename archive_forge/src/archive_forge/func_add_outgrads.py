from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def add_outgrads(prev_g_flagged, g):
    sparse = type(g) in sparse_object_types
    if prev_g_flagged:
        vs = vspace(g)
        prev_g, mutable = prev_g_flagged
        if mutable:
            if sparse:
                return (sparse_add(vs, prev_g, g), True)
            else:
                return (vs.mut_add(prev_g, g), True)
        elif sparse:
            prev_g_mutable = vs.mut_add(None, prev_g)
            return (sparse_add(vs, prev_g_mutable, g), True)
        else:
            return (vs.add(prev_g, g), True)
    elif sparse:
        return (sparse_add(vspace(g), None, g), True)
    else:
        return (g, False)