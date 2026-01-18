from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.utils._seq_dataset import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('dense_dataset,sparse_dataset', [(_make_dense_dataset(float_dtype), _make_sparse_dataset(csr_container, float_dtype)) for float_dtype, csr_container in product(floating, CSR_CONTAINERS)])
def test_seq_dataset_shuffle(dense_dataset, sparse_dataset):
    for i in range(5):
        _, _, _, idx1 = dense_dataset._next_py()
        _, _, _, idx2 = sparse_dataset._next_py()
        assert idx1 == i
        assert idx2 == i
    for i in [132, 50, 9, 18, 58]:
        _, _, _, idx1 = dense_dataset._random_py()
        _, _, _, idx2 = sparse_dataset._random_py()
        assert idx1 == i
        assert idx2 == i
    seed = 77
    dense_dataset._shuffle_py(seed)
    sparse_dataset._shuffle_py(seed)
    idx_next = [63, 91, 148, 87, 29]
    idx_shuffle = [137, 125, 56, 121, 127]
    for i, j in zip(idx_next, idx_shuffle):
        _, _, _, idx1 = dense_dataset._next_py()
        _, _, _, idx2 = sparse_dataset._next_py()
        assert idx1 == i
        assert idx2 == i
        _, _, _, idx1 = dense_dataset._random_py()
        _, _, _, idx2 = sparse_dataset._random_py()
        assert idx1 == j
        assert idx2 == j