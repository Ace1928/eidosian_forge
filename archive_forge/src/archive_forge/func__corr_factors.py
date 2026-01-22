import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
def _corr_factors(self):
    """correlation of factors implied by rotation

        If the rotation is oblique, then the factors are correlated.

        currently not cached

        Returns
        -------
        corr_f : ndarray
            correlation matrix of rotated factors, assuming initial factors are
            orthogonal
        """
    T = self.rotation_matrix
    corr_f = T.T.dot(T)
    return corr_f