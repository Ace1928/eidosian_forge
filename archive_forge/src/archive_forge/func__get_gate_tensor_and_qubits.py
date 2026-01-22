from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def _get_gate_tensor_and_qubits(gate: Gate) -> Tuple[np.ndarray, List[int]]:
    """Given a gate ``Instruction``, turn it into a matrix and extract qubit indices.

    :param gate: the instruction
    :return: tensor, qubit_inds.
    """
    if len(gate.params) > 0:
        matrix = QUANTUM_GATES[gate.name](*gate.params)
    else:
        matrix = QUANTUM_GATES[gate.name]
    qubit_inds = [q.index for q in gate.qubits]
    tensor = np.reshape(matrix, (2,) * len(qubit_inds) * 2)
    return (tensor, qubit_inds)