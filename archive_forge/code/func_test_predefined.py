from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('scipy', 'pygslodeiv2', 'pycvodes', 'pyodeint')
@pytest.mark.parametrize('solver', ['scipy', 'gsl', 'odeint', 'cvode'])
def test_predefined(solver):
    odes = ODESys(vdp_f, vdp_j, dfdx=vdp_dfdt)
    xout = [0, 0.7, 1.3, 2]
    yout, info = odes.predefined([1, 0], xout, params=[2.0], integrator=solver)
    assert np.allclose(yout[-1, :], [-1.89021896, -0.71633577])