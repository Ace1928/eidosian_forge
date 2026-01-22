from statsmodels.compat.pandas import testing as pdt
import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import MLEInfluence
def _check_looo(self):
    infl = self.infl1
    results = getattr(infl.results, '_results', infl.results)
    res_looo = infl._res_looo
    mask_infl = infl.cooks_distance[0] > 2 * infl.cooks_distance[0].std()
    mask_low = ~mask_infl
    diff_params = results.params - res_looo['params']
    assert_allclose(infl.d_params[mask_low], diff_params[mask_low], atol=0.05)
    assert_allclose(infl.params_one[mask_low], res_looo['params'][mask_low], rtol=0.01)