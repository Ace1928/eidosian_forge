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
def _get_csr_data_2(self, wire_order, coeff):
    """Computes the sparse matrix data of the Pauli word times a coefficient, given a wire order."""
    full_word = [self[wire] for wire in wire_order]
    nwords = len(full_word)
    if nwords < 2:
        return (np.array([1.0]), self._get_csr_data(wire_order, coeff))
    outer = self._get_csr_data(wire_order[:nwords // 2], 1.0)
    inner = self._get_csr_data(wire_order[nwords // 2:], coeff)
    return (outer, inner)