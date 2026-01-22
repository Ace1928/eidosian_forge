from collections import defaultdict, OrderedDict
from itertools import permutations
import math
import pytest
from chempy import Equilibrium, Reaction, ReactionSystem, Substance
from chempy.thermodynamics.expressions import MassActionEq
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.testing import requires
from .test_rates import _get_SpecialFraction_rsys
from ..arrhenius import ArrheniusParam
from ..rates import Arrhenius, MassAction, Radiolytic, RampedTemp
from .._rates import ShiftedTPoly
from ..ode import (
from ..integrated import dimerization_irrev, binary_rev
def _check_cstr(odesys, fr, fc, extra_pars=None):
    tout, c0 = (np.linspace(0, 0.13, 7), {'H2O2': 2, 'O2': 4, 'H2O': 3})
    params = {fr: 13, fc['H2O2']: 11, fc['O2']: 43, fc['H2O']: 45}
    params.update(extra_pars or {})
    res = odesys.integrate(tout, c0, params)
    from chempy.kinetics.integrated import binary_irrev_cstr

    def get_analytic(result, k, n):
        ref = binary_irrev_cstr(result.xout, 5, result.named_dep('H2O2')[0], result.named_dep(k)[0], result.named_param(fc['H2O2']), result.named_param(fc[k]), result.named_param(fr), n)
        return np.array(ref).T
    ref_O2 = get_analytic(res, 'O2', 1)
    ref_H2O = get_analytic(res, 'H2O', 2)
    assert np.allclose(res.named_dep('H2O2'), ref_O2[:, 0])
    assert np.allclose(res.named_dep('H2O2'), ref_H2O[:, 0])
    assert np.allclose(res.named_dep('O2'), ref_O2[:, 1])
    assert np.allclose(res.named_dep('H2O'), ref_H2O[:, 1])