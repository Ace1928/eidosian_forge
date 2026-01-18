import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2

        Summary of test results

        Parameters
        ----------
        show_contrast_L : bool
            Whether to show contrast_L matrix
        show_transform_M : bool
            Whether to show transform_M matrix
        show_constant_C : bool
            Whether to show the constant_C
        