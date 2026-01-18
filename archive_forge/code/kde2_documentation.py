from statsmodels.compat.python import lzip
import numpy as np
from statsmodels.tools.validation import array_like
from . import kernels

    Kernel Density Estimator

    Parameters
    ----------
    x : array_like
        N-dimensional array from which the density is to be estimated
    kernel : Kernel Class
        Should be a class from *
    