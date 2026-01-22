from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
def _decompose_b_gate_into_two_fsims(*, fsim_gate: 'cirq.FSimGate', qubits: Sequence['cirq.Qid']) -> List['cirq.Operation']:
    kak = linalg.kak_decomposition(_B)
    result = _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(qubits=qubits, fsim_gate=fsim_gate, canonical_x_kak_coefficient=kak.interaction_coefficients[0], canonical_y_kak_coefficient=kak.interaction_coefficients[1])
    return list(_fix_single_qubit_gates_around_kak_interaction(desired=kak, qubits=qubits, operations=result))