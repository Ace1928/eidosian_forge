from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
def _noisy_moment_impl_moments(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
    return self.noisy_moments([moment], system_qubits)