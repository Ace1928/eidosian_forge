from typing import List, TYPE_CHECKING
import functools
import numpy as np
from cirq import ops, protocols, qis, sim
def _SWAP(q1: int, q2: int, args: sim.CliffordTableauSimulationState, operations: List[ops.Operation], qubits: List['cirq.Qid']):
    protocols.act_on(ops.SWAP, args, qubits=[qubits[q1], qubits[q2]], allow_decompose=False)
    operations.append(ops.SWAP(qubits[q1], qubits[q2]))