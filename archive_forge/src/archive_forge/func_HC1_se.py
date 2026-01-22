from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
@cache_readonly
def HC1_se(self):
    """
        MacKinnon and White's (1985) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as sqrt(diag(n/(n-p)*HC_0).

        When HC1_se or cov_HC1 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        n/(n-p)*resid**2.
        """
    return np.sqrt(np.diag(self.cov_HC1))