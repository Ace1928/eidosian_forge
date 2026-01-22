import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
class MyK(Expr):
    argument_names = ('H', 'S')
    parameter_keys = ('T',)
    R = 8.3145

    def __call__(self, variables, backend=math):
        H, S = self.all_args(variables, backend=backend)
        T, = self.all_params(variables, backend=backend)
        return backend.exp(-(H - T * S) / (self.R * T))