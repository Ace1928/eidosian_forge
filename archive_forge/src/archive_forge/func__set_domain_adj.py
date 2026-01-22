import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def _set_domain_adj(self):
    """ Adjust the domain based on loc and scale. """
    loc = self.loc
    scale = self.scale
    lb = self._domain[0] * scale + loc
    ub = self._domain[1] * scale + loc
    self._domain_adj = (lb, ub)