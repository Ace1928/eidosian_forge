from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
def _simultaneous_ci(self):
    """Compute simultaneous confidence intervals for comparison of means.
        """
    self.halfwidths = simultaneous_ci(self.q_crit, self.variance, self._multicomp.groupstats.groupnobs, self._multicomp.pairindices)