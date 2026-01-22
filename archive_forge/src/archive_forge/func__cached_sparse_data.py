import warnings
from copy import copy
from functools import reduce, lru_cache
from typing import Iterable
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
@lru_cache
def _cached_sparse_data(op):
    """Returns the sparse data and indices of a Pauli operator."""
    if op == 'I':
        data = np.array([1.0, 1.0], dtype=np.complex128)
        indices = np.array([0, 1], dtype=np.int64)
    elif op == 'X':
        data = np.array([1.0, 1.0], dtype=np.complex128)
        indices = np.array([1, 0], dtype=np.int64)
    elif op == 'Y':
        data = np.array([-1j, 1j], dtype=np.complex128)
        indices = np.array([1, 0], dtype=np.int64)
    elif op == 'Z':
        data = np.array([1.0, -1.0], dtype=np.complex128)
        indices = np.array([0, 1], dtype=np.int64)
    return (data, indices)