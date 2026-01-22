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
def _prove_html(claim, show=False, **keywords):
    """Version of function `prove` that renders HTML."""
    if z3_debug():
        _z3_assert(is_bool(claim), 'Z3 Boolean expression expected')
    s = Solver()
    s.set(**keywords)
    s.add(Not(claim))
    if show:
        print(s)
    r = s.check()
    if r == unsat:
        print('<b>proved</b>')
    elif r == unknown:
        print('<b>failed to prove</b>')
        print(s.model())
    else:
        print('<b>counterexample</b>')
        print(s.model())