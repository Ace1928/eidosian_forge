import abc
from typing import Any, cast, Generic, Optional, Sequence, TYPE_CHECKING, TypeVar, Union
import numpy as np
import sympy
from cirq import linalg, ops, protocols
from cirq.ops import common_gates, global_phase_op, matrix_gates, swap_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.simulation_state import SimulationState
from cirq.type_workarounds import NotImplementedType
def _strat_apply_mixture(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
    mixture = protocols.mixture(val, None)
    if mixture is None:
        return NotImplemented
    if not all((linalg.is_unitary(m) for _, m in mixture)):
        return NotImplemented
    probabilities, unitaries = zip(*mixture)
    index = self.prng.choice(len(unitaries), p=probabilities)
    return self._strat_act_from_single_qubit_decompose(matrix_gates.MatrixGate(unitaries[index]), qubits)