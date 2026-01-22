import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
Sampling of random variates

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of random variates to be generated (default is 1).

        Returns
        -------
        rvs : ndarray
            The random variates distributed according to the probability
            distribution defined by the pdf.

        