from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
def vdp_j(t, y, p):
    Jmat = np.zeros((2, 2))
    Jmat[0, 0] = 0
    Jmat[0, 1] = 1
    Jmat[1, 0] = -1 - p[0] * 2 * y[1] * y[0]
    Jmat[1, 1] = p[0] * (1 - y[0] ** 2)
    return Jmat