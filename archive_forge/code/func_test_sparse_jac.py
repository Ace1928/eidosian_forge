from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('pycvodes', 'scipy')
@pycvodes_klu
def test_sparse_jac():
    odesys = ODESys(sine, sine_jac_sparse, nnz=2)
    A, k = (2, 3)
    xout, yout, info = odesys.integrate(np.linspace(0, 1), [0, A * k], [k], integrator='cvode', linear_solver='klu')
    assert info['success']
    ref = [A * np.sin(k * (xout - xout[0])), A * np.cos(k * (xout - xout[0])) * k]
    assert np.allclose(yout[:, 0], ref[0], atol=1e-05, rtol=1e-05)
    assert np.allclose(yout[:, 1], ref[1], atol=1e-05, rtol=1e-05)