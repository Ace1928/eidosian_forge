import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def _compute_dispersion(self, data):
    """
        Computes the measure of dispersion.

        The minimum of the standard deviation and interquartile range / 1.349

        Notes
        -----
        Reimplemented in `KernelReg`, because the first column of `data` has to
        be removed.

        References
        ----------
        See the user guide for the np package in R.
        In the notes on bwscaling option in npreg, npudens, npcdens there is
        a discussion on the measure of dispersion
        """
    return _compute_min_std_IQR(data)