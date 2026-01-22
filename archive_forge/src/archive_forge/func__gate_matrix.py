from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def _gate_matrix(gate: Gate) -> np.ndarray:
    if len(gate.modifiers) == 0:
        if len(gate.params) > 0:
            return QUANTUM_GATES[gate.name](*gate.params)
        else:
            return QUANTUM_GATES[gate.name]
    else:
        mod = gate.modifiers[0]
        if mod == 'DAGGER':
            child = _strip_modifiers(gate, limit=1)
            return _gate_matrix(child).conj().T
        elif mod == 'CONTROLLED':
            child = _strip_modifiers(gate, limit=1)
            matrix = _gate_matrix(child)
            return np.kron(zero, np.eye(*matrix.shape)) + np.kron(one, matrix)
        elif mod == 'FORKED':
            assert len(gate.params) % 2 == 0
            p0, p1 = (gate.params[:len(gate.params) // 2], gate.params[len(gate.params) // 2:])
            child = _strip_modifiers(gate, limit=1)
            child.params = p0
            mat0 = _gate_matrix(child)
            child.params = p1
            mat1 = _gate_matrix(child)
            return np.kron(zero, mat0) + np.kron(one, mat1)
        else:
            raise TypeError('Unsupported gate modifier {}'.format(mod))