import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def _compare_sparse_coo_values(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool) -> None:
    """Compares sparse COO tensors by comparing

        - the number of sparse dimensions,
        - the number of non-zero elements (nnz) for equality,
        - the indices for equality, and
        - the values for closeness.
        """
    if actual.sparse_dim() != expected.sparse_dim():
        self._fail(AssertionError, f'The number of sparse dimensions in sparse COO tensors does not match: {actual.sparse_dim()} != {expected.sparse_dim()}')
    if actual._nnz() != expected._nnz():
        self._fail(AssertionError, f'The number of specified values in sparse COO tensors does not match: {actual._nnz()} != {expected._nnz()}')
    self._compare_regular_values_equal(actual._indices(), expected._indices(), identifier='Sparse COO indices')
    self._compare_regular_values_close(actual._values(), expected._values(), rtol=rtol, atol=atol, equal_nan=equal_nan, identifier='Sparse COO values')