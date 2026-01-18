import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (

    Load data with deterministic terms in `exog_coint`.

    Same as load_results_statsmodels() except that deterministic terms inside
    the cointegration relation are provided to :class:`VECM`'s `__init__()`
    method via the `eoxg_coint` parameter. This is to check whether the same
    results are produced no matter whether `exog_coint` or the `deterministic`
    argument is being used.

    Parameters
    ----------
    dataset : DataSet
    