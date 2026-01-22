import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
class GibbsExpr(Expr):
    parameter_keys = ('temperature',)
    argument_names = tuple('dS_over_R dCp_over_R dH_over_R Tref'.split())

    def __call__(self, variables, backend=patched_numpy, **kwargs):
        am = dict(zip(self.argument_names, map(simplified, self.all_args(variables, backend=backend))))
        T, = self.all_params(variables, backend=backend)
        return backend.exp(am['dS_over_R']) * (T / am['Tref']) ** am['dCp_over_R'] * backend.exp(-am['dH_over_R'] / T)