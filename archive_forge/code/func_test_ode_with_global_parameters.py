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
@requires('pyodesys')
def test_ode_with_global_parameters():
    ratex = MassAction(Arrhenius([10000000000.0, 40000.0 / 8.3145]))
    rxn = Reaction({'A': 1}, {'B': 1}, ratex)
    rsys = ReactionSystem([rxn], 'A B')
    odesys, extra = get_odesys(rsys, include_params=False)
    param_keys, unique_keys, p_units = map(extra.get, 'param_keys unique p_units'.split())
    conc = {'A': 3, 'B': 5}
    x, y, p = odesys.to_arrays(-37, conc, {'temperature': 298.15})
    fout = odesys.f_cb(x, y, p)
    ref = 3 * 10000000000.0 * np.exp(-40000.0 / 8.3145 / 298.15)
    assert np.all(abs((fout[:, 0] + ref) / ref) < 1e-14)
    assert np.all(abs((fout[:, 1] - ref) / ref) < 1e-14)