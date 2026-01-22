import dataclasses
import functools
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import sympy
from cirq import devices, ops, protocols, qis
from cirq._import import LazyLoader
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
def _right_mul(mat: np.ndarray) -> np.ndarray:
    """Superoperator associated with right multiplication by a square matrix."""
    mat = np.asarray(mat)
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError(f'_right_mul only accepts square matrices, but input matrix has shape {mat.shape}.')
    dim = mat.shape[-1]
    return np.kron(np.eye(dim), np.swapaxes(mat, -2, -1))