import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
class Eyring4(Expr):
    nargs = 4

    def __call__(self, variables, reaction, backend=math):
        Sact_fact, Hact_exp, Sref_fact, Href_exp = self.all_args(variables, backend=backend)
        T = variables['temperature']
        return T * Sact_fact / Sref_fact * backend.exp((Href_exp - Hact_exp) / T)