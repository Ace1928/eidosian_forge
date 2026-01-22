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
@staticmethod
def _reshape_single_constant_tensor(v: sp.csc_matrix, lin_op_shape: tuple[int, int]) -> sp.csc_matrix:
    """
        Given v, which is a matrix of shape (p * lin_op_shape[0] * lin_op_shape[1], 1),
        reshape v into a matrix of shape (p * lin_op_shape[0], lin_op_shape[1]).
        """
    assert v.shape[1] == 1
    p = np.prod(v.shape) // np.prod(lin_op_shape)
    old_shape = (v.shape[0] // p, v.shape[1])
    coo = v.tocoo()
    data, stacked_rows = (coo.data, coo.row)
    slices, rows = np.divmod(stacked_rows, old_shape[0])
    new_cols, new_rows = np.divmod(rows, lin_op_shape[0])
    new_rows = slices * lin_op_shape[0] + new_rows
    new_stacked_shape = (p * lin_op_shape[0], lin_op_shape[1])
    return sp.csc_matrix((data, (new_rows, new_cols)), shape=new_stacked_shape)