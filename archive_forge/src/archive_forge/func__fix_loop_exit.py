import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def _fix_loop_exit(cfg, loop):
    """
    Fixes loop.exits for Py3.8+ bytecode CFG changes.
    This is to handle `break` inside loops.
    """
    postdoms = cfg.post_dominators()
    exits = reduce(operator.and_, [postdoms[b] for b in loop.exits], loop.exits)
    if exits:
        body = loop.body | loop.exits - exits
        return loop._replace(exits=exits, body=body)
    else:
        return loop