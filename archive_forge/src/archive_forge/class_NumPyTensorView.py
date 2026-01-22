from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
class NumPyTensorView(DictTensorView):

    @property
    def rows(self) -> int:
        """
        Number of rows of the TensorView.
        This is the second dimension of the 3d tensor.
        """
        if self.tensor is not None:
            return next(iter(next(iter(self.tensor.values())).values())).shape[1]
        else:
            raise ValueError

    def get_tensor_representation(self, row_offset: int, total_rows: int) -> TensorRepresentation:
        """
        Returns a TensorRepresentation of [A b] tensor.
        This function iterates through all the tensor data and constructs the
        respective representation in COO format. To obtain the data, the tensor must be
        flattened as it is not in a sparse format. The row and column indices are obtained
        with numpy tiling/repeating along with their respective offsets.

        Note: CVXPY currently only supports usage of sparse matrices after the canonicalization.
        Therefore, we must return tensor representations in a (data, (row,col)) format.
        This could be changed once dense matrices are accepted.
        """
        assert self.tensor is not None
        shape = (total_rows, self.var_length + 1)
        tensor_representations = []
        for variable_id, variable_tensor in self.tensor.items():
            for parameter_id, parameter_tensor in variable_tensor.items():
                param_size, rows, cols = parameter_tensor.shape
                tensor_representations.append(TensorRepresentation(parameter_tensor.flatten(order='F'), np.repeat(np.tile(np.arange(rows), cols), param_size) + row_offset, np.repeat(np.repeat(np.arange(cols), rows), param_size) + self.id_to_col[variable_id], np.tile(np.arange(param_size), rows * cols) + self.param_to_col[parameter_id], shape=shape))
        return TensorRepresentation.combine(tensor_representations)

    def select_rows(self, rows: np.ndarray) -> None:
        """
        Select 'rows' from tensor.
        The rows of the 3d tensor are in axis=1, this function selects a subset
        of the original tensor.
        """

        def func(x):
            return x[:, rows, :]
        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        """
        Apply 'func' across all variables and parameter slices.
        The tensor functions in the NumPyBackend manipulate 3d arrays.
        Therefore, this function applies 'func' directly to the tensor 'v'.
        """
        self.tensor = {var_id: {k: func(v) for k, v in parameter_repr.items()} for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any, is_parameter_free: bool) -> NumPyTensorView:
        """
        Create new NumPyTensorView with same shape information as self,
        but new tensor data.
        """
        return NumPyTensorView(variable_ids, tensor, is_parameter_free, self.param_size_plus_one, self.id_to_col, self.param_to_size, self.param_to_col, self.var_length)

    @staticmethod
    def apply_to_parameters(func: Callable, parameter_representation: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """
        Apply 'func' to the entire tensor of the parameter representation.
        """
        return {k: func(v) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Apply element-wise addition on two dense numpy arrays
        """
        return a + b

    @staticmethod
    def tensor_type():
        """
        The tensor is represented as a 3-dimensional dense numpy array
        """
        return np.ndarray