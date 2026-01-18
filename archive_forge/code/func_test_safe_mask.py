import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_safe_mask(csr_container):
    random_state = check_random_state(0)
    X = random_state.rand(5, 4)
    X_csr = csr_container(X)
    mask = [False, False, True, True, True]
    mask = safe_mask(X, mask)
    assert X[mask].shape[0] == 3
    mask = safe_mask(X_csr, mask)
    assert X_csr[mask].shape[0] == 3