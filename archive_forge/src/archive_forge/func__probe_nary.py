from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def _probe_nary(f, args, ctx):
    if z3_debug():
        _z3_assert(len(args) > 0, 'At least one argument expected')
    num = len(args)
    r = _to_probe(args[0], ctx)
    for i in range(num - 1):
        r = Probe(f(ctx.ref(), r.probe, _to_probe(args[i + 1], ctx).probe), ctx)
    return r