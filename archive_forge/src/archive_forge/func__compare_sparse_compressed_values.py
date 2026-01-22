import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def _compare_sparse_compressed_values(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool) -> None:
    """Compares sparse compressed tensors by comparing

        - the number of non-zero elements (nnz) for equality,
        - the plain indices for equality,
        - the compressed indices for equality, and
        - the values for closeness.
        """
    format_name, compressed_indices_method, plain_indices_method = {torch.sparse_csr: ('CSR', torch.Tensor.crow_indices, torch.Tensor.col_indices), torch.sparse_csc: ('CSC', torch.Tensor.ccol_indices, torch.Tensor.row_indices), torch.sparse_bsr: ('BSR', torch.Tensor.crow_indices, torch.Tensor.col_indices), torch.sparse_bsc: ('BSC', torch.Tensor.ccol_indices, torch.Tensor.row_indices)}[actual.layout]
    if actual._nnz() != expected._nnz():
        self._fail(AssertionError, f'The number of specified values in sparse {format_name} tensors does not match: {actual._nnz()} != {expected._nnz()}')
    actual_compressed_indices = compressed_indices_method(actual)
    expected_compressed_indices = compressed_indices_method(expected)
    indices_dtype = torch.promote_types(actual_compressed_indices.dtype, expected_compressed_indices.dtype)
    self._compare_regular_values_equal(actual_compressed_indices.to(indices_dtype), expected_compressed_indices.to(indices_dtype), identifier=f'Sparse {format_name} {compressed_indices_method.__name__}')
    self._compare_regular_values_equal(plain_indices_method(actual).to(indices_dtype), plain_indices_method(expected).to(indices_dtype), identifier=f'Sparse {format_name} {plain_indices_method.__name__}')
    self._compare_regular_values_close(actual.values(), expected.values(), rtol=rtol, atol=atol, equal_nan=equal_nan, identifier=f'Sparse {format_name} values')