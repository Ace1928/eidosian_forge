from functools import lru_cache, reduce, singledispatch
from itertools import product
from typing import List, Union
from warnings import warn
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.tape import OperationRecorder
from pennylane.wires import Wires
@lru_cache
def _get_pauli_map(n):
    """Return a list of Pauli operator objects acting on wires `0` up to `n`.

    This function is used to accelerate ``qchem.observable_hf.jordan_wigner``.
    """
    warn('_get_pauli_map is deprecated, as it is no longer used.', qml.PennyLaneDeprecationWarning)
    return [{'I': qml.Identity(i), 'X': qml.X(i), 'Y': qml.Y(i), 'Z': qml.Z(i)} for i in range(n + 1)]