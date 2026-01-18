import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
@requires('numpy')
def test_specialfraction_rateexpr():
    rsys = _get_SpecialFraction_rsys(11, 13)
    r = rsys.rxns[0]
    conc = {'H2': 2, 'Br2': 3, 'HBr': 5}

    def _check(k, kprime, c):
        ref = k * c['H2'] * c['Br2'] ** 1.5 / (c['Br2'] + kprime * c['HBr'])
        assert abs(r.rate_expr()(c) - ref) < 1e-15
    _check(11, 13, conc)