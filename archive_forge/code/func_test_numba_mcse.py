import importlib
import numpy as np
import pytest
from ...data import load_arviz_data
from ...rcparams import rcParams
from ...stats import bfmi, mcse, rhat
from ...stats.diagnostics import _mc_error, ks_summary
from ...utils import Numba
from ..helpers import running_on_ci
from .test_diagnostics import data  # pylint: disable=unused-import
@pytest.mark.parametrize('method', ('mean', 'sd', 'quantile'))
def test_numba_mcse(method, prob=None):
    """Numba test for mcse."""
    state = Numba.numba_flag
    school = np.random.rand(100, 100)
    if method == 'quantile':
        prob = 0.8
    Numba.disable_numba()
    non_numba = mcse(school, method=method, prob=prob)
    Numba.enable_numba()
    with_numba = mcse(school, method=method, prob=prob)
    assert np.allclose(with_numba, non_numba)
    assert Numba.numba_flag == state