from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.utils._seq_dataset import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSR_CONTAINERS
def _make_dense_dataset(float_dtype):
    if float_dtype == np.float32:
        return ArrayDataset32(X32, y32, sample_weight32, seed=42)
    return ArrayDataset64(X64, y64, sample_weight64, seed=42)