import importlib
from unittest.mock import Mock
import numpy as np
import pytest
from ...stats.stats_utils import stats_variance_2d as svar
from ...utils import Numba, _numba_var, numba_check
from ..helpers import running_on_ci
from .test_utils import utils_with_numba_import_fail  # pylint: disable=unused-import
def test_utils_fixture(utils_with_numba_import_fail):
    """Test of utils fixture to ensure mock is applied correctly"""
    import numba
    with pytest.raises(ImportError):
        utils_with_numba_import_fail.importlib.import_module('numba')