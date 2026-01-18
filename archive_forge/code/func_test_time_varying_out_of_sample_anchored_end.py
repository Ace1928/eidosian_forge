from statsmodels.compat.pandas import MONTH_END
import warnings
import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas as pd
import pytest
from scipy.stats import ortho_group
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.statespace import (
from statsmodels.tsa.vector_ar.tests.test_var import get_macrodata
def test_time_varying_out_of_sample_anchored_end(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    with pytest.raises(ValueError, match='Model has time-varying'):
        mod.impulse_responses([], steps=2, anchor='end')
    new_Z = np.random.normal(size=mod['design', :, :, -2:].shape)
    new_T = np.random.normal(size=mod['transition', :, :, -2:].shape)
    irfs = mod.impulse_responses([], steps=2, anchor='end', design=new_Z, transition=new_T)
    cirfs = mod.impulse_responses([], steps=2, anchor='end', design=new_Z, transition=new_T, cumulative=True)
    oirfs = mod.impulse_responses([], steps=2, anchor='end', design=new_Z, transition=new_T, orthogonalized=True)
    coirfs = mod.impulse_responses([], steps=2, anchor='end', design=new_Z, transition=new_T, cumulative=True, orthogonalized=True)
    R = mod['selection']
    Q = mod['state_cov', ..., -1]
    L = np.linalg.cholesky(Q)
    desired_irfs = np.zeros((2, 2)) * np.nan
    desired_oirfs = np.zeros((2, 2)) * np.nan
    tmp = R[..., -1]
    desired_irfs[0] = new_Z[:, :, 0].dot(tmp)[:, 0]
    desired_oirfs[0] = new_Z[:, :, 0].dot(tmp).dot(L)[:, 0]
    tmp = new_T[..., 0].dot(tmp)
    desired_irfs[1] = new_Z[:, :, 1].dot(tmp)[:, 0]
    desired_oirfs[1] = new_Z[:, :, 1].dot(tmp).dot(L)[:, 0]
    assert_allclose(irfs, desired_irfs)
    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))