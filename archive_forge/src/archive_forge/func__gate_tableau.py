from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def _gate_tableau(num_qubits: int, gate: raw_types.Gate) -> 'cirq.CliffordTableau':
    qubits = devices.LineQubit.range(num_qubits)
    t = qis.CliffordTableau(num_qubits=num_qubits)
    args = sim.CliffordTableauSimulationState(tableau=t, qubits=qubits, prng=np.random.RandomState())
    protocols.act_on(gate, args, qubits, allow_decompose=False)
    return args.tableau