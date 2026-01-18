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
def test_numba_bfmi():
    """Numba test for bfmi."""
    state = Numba.numba_flag
    school = load_arviz_data('centered_eight')
    data_md = np.random.rand(100, 100, 10)
    Numba.disable_numba()
    non_numba = bfmi(school.posterior['mu'].values)
    non_numba_md = bfmi(data_md)
    Numba.enable_numba()
    with_numba = bfmi(school.posterior['mu'].values)
    with_numba_md = bfmi(data_md)
    assert np.allclose(non_numba_md, with_numba_md)
    assert np.allclose(with_numba, non_numba)
    assert state == Numba.numba_flag