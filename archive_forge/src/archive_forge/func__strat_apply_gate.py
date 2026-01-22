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
def _strat_apply_gate(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
    if not protocols.has_stabilizer_effect(val):
        return NotImplemented
    gate = val.gate if isinstance(val, ops.Operation) else val
    axes = self.get_axes(qubits)
    exponent = cast(float, getattr(gate, 'exponent', None))
    if isinstance(gate, common_gates.XPowGate):
        self._state.apply_x(axes[0], exponent, gate.global_shift)
    elif isinstance(gate, common_gates.YPowGate):
        self._state.apply_y(axes[0], exponent, gate.global_shift)
    elif isinstance(gate, common_gates.ZPowGate):
        self._state.apply_z(axes[0], exponent, gate.global_shift)
    elif isinstance(gate, common_gates.HPowGate):
        self._state.apply_h(axes[0], exponent, gate.global_shift)
    elif isinstance(gate, common_gates.CXPowGate):
        self._state.apply_cx(axes[0], axes[1], exponent, gate.global_shift)
    elif isinstance(gate, common_gates.CZPowGate):
        self._state.apply_cz(axes[0], axes[1], exponent, gate.global_shift)
    elif isinstance(gate, global_phase_op.GlobalPhaseGate):
        if isinstance(gate.coefficient, sympy.Expr):
            return NotImplemented
        self._state.apply_global_phase(gate.coefficient)
    elif isinstance(gate, swap_gates.SwapPowGate):
        self._swap(axes[0], axes[1], exponent, gate.global_shift)
    else:
        return NotImplemented
    return True