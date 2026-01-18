from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('scipy')
def test_zero_time_adaptive():
    odes = ODESys(sine, sine_jac)
    xout, yout, info = odes.integrate(0, [0, 1], [2])
    assert xout.shape == (1,)
    assert yout.shape == (1, 2)