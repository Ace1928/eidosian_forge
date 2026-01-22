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
class CanonBackend(ABC):

    def __init__(self, id_to_col: dict[int, int], param_to_size: dict[int, int], param_to_col: dict[int, int], param_size_plus_one: int, var_length: int):
        """
        CanonBackend handles the compilation from LinOp trees to a final sparse tensor through its
        subclasses.

        Parameters
        ----------
        id_to_col: mapping of variable id to column offset in A.
        param_to_size: mapping of parameter id to the corresponding number of elements.
        param_to_col: mapping of parameter id to the offset in axis 2.
        param_size_plus_one: integer representing shape[2], i.e., the number of slices along axis 2
                             plus_one refers to the non-parametrized slice of the tensor.
        var_length: number of columns in A.
        """
        self.param_size_plus_one = param_size_plus_one
        self.id_to_col = id_to_col
        self.param_to_size = param_to_size
        self.param_to_col = param_to_col
        self.var_length = var_length

    @classmethod
    def get_backend(cls, backend_name: str, *args, **kwargs) -> CanonBackend:
        """
        Map the name of a subclass and its initializing arguments to an instance of the subclass.

        Parameters
        ----------
        backend_name: key pointing to the subclass.
        args: Arguments required to initialize the subclass.

        Returns
        -------
        Initialized CanonBackend subclass.
        """
        backends = {NUMPY_CANON_BACKEND: NumPyCanonBackend, SCIPY_CANON_BACKEND: SciPyCanonBackend, RUST_CANON_BACKEND: RustCanonBackend}
        return backends[backend_name](*args, **kwargs)

    @abstractmethod
    def build_matrix(self, lin_ops: list[LinOp]) -> sp.csc_matrix:
        """
        Main function called from canonInterface.
        Given a list of LinOp trees, each representing a constraint (or the objective), get the
        [A b] Tensor for each, stack them and return the result reshaped as a 2D sp.csc_matrix
        of shape (total_rows * (var_length + 1)), param_size_plus_one)

        Parameters
        ----------
        lin_ops: list of linOp trees.

        Returns
        -------
        2D sp.csc_matrix representing the constraints (or the objective).
        """
        pass