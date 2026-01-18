import numpy as np
from numpy.testing import assert_, assert_allclose, assert_raises
import statsmodels.datasets.macrodata.data as macro
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.var_model import VAR
from .JMulTi_results.parse_jmulti_var_output import (

    Translate seasons to exog matrix.

    Parameters
    ----------
    seasons : int
        Number of seasons.
    endog_len : int
        Number of observations.

    Returns
    -------
    exog : ndarray or None
        If seasonal deterministic terms exist, the corresponding exog-matrix is
        returned.
        Otherwise, None is returned.
    