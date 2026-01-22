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
def RealVar(idx, ctx=None):
    """
    Create a real free variable. Free variables are used to create quantified formulas.
    They are also used to create polynomials.

    >>> RealVar(0)
    Var(0)
    """
    return Var(idx, RealSort(ctx))