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
def _wire_map_from_pauli_pair(pauli_word_1, pauli_word_2):
    """Generate a wire map from the union of wires of two Paulis.

    Args:
        pauli_word_1 (.Operation): A Pauli word.
        pauli_word_2 (.Operation): A second Pauli word.

    Returns:
        dict[Union[str, int], int]): dictionary containing all wire labels used
        in the Pauli word as keys, and unique integer labels as their values.
    """
    wire_labels = Wires.all_wires([pauli_word_1.wires, pauli_word_2.wires]).labels
    return {label: i for i, label in enumerate(wire_labels)}