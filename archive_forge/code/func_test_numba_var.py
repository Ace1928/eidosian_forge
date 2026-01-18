import importlib
from unittest.mock import Mock
import numpy as np
import pytest
from ...stats.stats_utils import stats_variance_2d as svar
from ...utils import Numba, _numba_var, numba_check
from ..helpers import running_on_ci
from .test_utils import utils_with_numba_import_fail  # pylint: disable=unused-import
@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('ddof', (0, 1))
def test_numba_var(axis, ddof):
    """Method to test numba_var."""
    flag = Numba.numba_flag
    data_1 = np.random.randn(100, 100)
    data_2 = np.random.rand(100)
    with_numba_1 = _numba_var(svar, np.var, data_1, axis=axis, ddof=ddof)
    with_numba_2 = _numba_var(svar, np.var, data_2, ddof=ddof)
    Numba.disable_numba()
    non_numba_1 = _numba_var(svar, np.var, data_1, axis=axis, ddof=ddof)
    non_numba_2 = _numba_var(svar, np.var, data_2, ddof=ddof)
    Numba.enable_numba()
    assert flag == Numba.numba_flag
    assert np.allclose(with_numba_1, non_numba_1)
    assert np.allclose(with_numba_2, non_numba_2)