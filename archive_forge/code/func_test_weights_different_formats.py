import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
@pytest.mark.parametrize('formatted', [weights, np.asarray(weights), pd.Series(weights)], ids=['list', 'ndarray', 'Series'])
def test_weights_different_formats(formatted):
    check_weights_as_formats(formatted)