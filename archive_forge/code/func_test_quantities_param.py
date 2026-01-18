from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('quantities')
def test_quantities_param():
    import quantities as pq
    odesys = ODESys(sine, sine_jac, param_names=['k'], par_by_name=True, to_arrays_callbacks=(None, None, lambda p: [p[0].rescale(1 / pq.s).magnitude]))
    A, k = (2, 3)
    xout, yout, info = odesys.integrate(np.linspace(0, 1), [0, A * k], {'k': k / pq.second})
    assert info['success']
    assert xout.size > 7
    ref = [A * np.sin(k * (xout - xout[0])), A * np.cos(k * (xout - xout[0])) * k]
    assert np.allclose(yout[:, 0], ref[0], atol=1e-05, rtol=1e-05)
    assert np.allclose(yout[:, 1], ref[1], atol=1e-05, rtol=1e-05)