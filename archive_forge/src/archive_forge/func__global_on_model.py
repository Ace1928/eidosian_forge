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
def _global_on_model(ctx):
    fn, mdl = _on_models[ctx]
    fn(mdl)