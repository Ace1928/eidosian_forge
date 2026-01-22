import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def _set_bw_bounds(self, bw):
    """
        Sets bandwidth lower bound to effectively zero )1e-10), and for
        discrete values upper bound to 1.
        """
    bw[bw < 0] = 1e-10
    _, ix_ord, ix_unord = _get_type_pos(self.data_type)
    bw[ix_ord] = np.minimum(bw[ix_ord], 1.0)
    bw[ix_unord] = np.minimum(bw[ix_unord], 1.0)
    return bw