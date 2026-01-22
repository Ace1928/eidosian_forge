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
class MyK2(Expr):
    argument_names = ('H', 'S', 'Cp', 'Tref')
    argument_defaults = (0, 298.15)
    parameter_keys = 'T'
    R = 8.3145

    def __call__(self, variables, backend=math):
        H, S, Cp, Tref = self.all_args(variables, backend=backend)
        T, = self.all_params(variables, backend=backend)
        _H = H + Cp * (T - Tref)
        _S = S + Cp * backend.log(T / Tref)
        return backend.exp(-(_H - T * _S) / (self.R * T))