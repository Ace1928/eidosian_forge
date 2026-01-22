from typing import Any, FrozenSet, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def _fredkin(qubits: Sequence[cirq.Qid], context: cirq.DecompositionContext) -> cirq.OP_TREE:
    """Decomposition with 7 T and 10 clifford operations from https://arxiv.org/abs/1308.4134"""
    c, t1, t2 = qubits
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.CNOT(c, t1), cirq.H(t2)]
    yield [cirq.T(c), cirq.T(t1) ** (-1), cirq.T(t2)]
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.CNOT(c, t2), cirq.T(t1)]
    yield [cirq.CNOT(c, t1), cirq.T(t2) ** (-1)]
    yield [cirq.T(t1) ** (-1), cirq.CNOT(c, t2)]
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.T(t1), cirq.H(t2)]
    yield [cirq.CNOT(t2, t1)]