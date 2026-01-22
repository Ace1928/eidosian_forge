from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
def _noisy_moments_impl_moment(self, moments: Iterable['cirq.Moment'], system_qubits: Sequence['cirq.Qid']) -> Sequence['cirq.OP_TREE']:
    result = []
    for moment in moments:
        result.append(self.noisy_moment(moment, system_qubits))
    return result