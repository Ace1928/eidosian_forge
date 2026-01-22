from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
def _fit_collinear(self, atol=1e-14, rtol=1e-13, **kwds):
    """experimental, fit of the model without collinear variables

        This currently uses QR to drop variables based on the given
        sequence.
        Options will be added in future, when the supporting functions
        to identify collinear variables become available.
        """
    x = self.exog
    tol = atol + rtol * x.var(0)
    r = np.linalg.qr(x, mode='r')
    mask = np.abs(r.diagonal()) < np.sqrt(tol)
    idx_keep = np.where(~mask)[0]
    return self._fit_zeros(keep_index=idx_keep, **kwds)