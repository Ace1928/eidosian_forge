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
def _strat_decompose(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
    gate = val.gate if isinstance(val, ops.Operation) else val
    operations = protocols.decompose_once_with_qubits(gate, qubits, None)
    if operations is None or not all((protocols.has_stabilizer_effect(op) for op in operations)):
        return NotImplemented
    for op in operations:
        protocols.act_on(op, self)
    return True