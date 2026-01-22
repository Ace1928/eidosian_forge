import abc
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._doc import document
@dataclass(frozen=True)
class ProductState:
    """A quantum state that is a tensor product of one qubit states.

    For example, the |00⟩ state is `cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1)`.
    The |+⟩ state is a length-1 tensor product state and can be constructed
    with `cirq.KET_PLUS(q0)`.
    """
    states: Dict['cirq.Qid', _NamedOneQubitState]

    def __init__(self, states=None):
        if states is None:
            states = dict()
        object.__setattr__(self, 'states', states)

    @property
    def qubits(self) -> Sequence['cirq.Qid']:
        return sorted(self.states.keys())

    def __mul__(self, other: 'cirq.ProductState') -> 'cirq.ProductState':
        if not isinstance(other, ProductState):
            raise ValueError('Multiplication is only supported with other TensorProductStates.')
        dupe_qubits = set(other.states.keys()) & set(self.states.keys())
        if len(dupe_qubits) != 0:
            raise ValueError(f'You tried to tensor two states, but both contain factors for these qubits: {sorted(dupe_qubits)}')
        new_states = self.states.copy()
        new_states.update(other.states)
        return ProductState(new_states)

    def __str__(self) -> str:
        return ' * '.join((f'{st}({q})' for q, st in self.states.items()))

    def __repr__(self) -> str:
        states_dict_repr = ', '.join((f'{repr(key)}: {repr(val)}' for key, val in self.states.items()))
        return f'cirq.ProductState({{{states_dict_repr}}})'

    def __getitem__(self, qubit: 'cirq.Qid') -> _NamedOneQubitState:
        """Return the _NamedOneQubitState at the given qubit."""
        return self.states[qubit]

    def __iter__(self) -> Iterator[Tuple['cirq.Qid', _NamedOneQubitState]]:
        yield from self.states.items()

    def __len__(self) -> int:
        return len(self.states)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ProductState):
            return False
        return self.states == other.states

    def __hash__(self):
        return hash(tuple(self.states.items()))

    def _json_dict_(self):
        return {'states': list(self.states.items())}

    @classmethod
    def _from_json_dict_(cls, states, **kwargs):
        return cls(states=dict(states))

    def state_vector(self, qubit_order: Optional['cirq.QubitOrder']=None) -> np.ndarray:
        """The state-vector representation of this state."""
        from cirq import ops
        if qubit_order is None:
            qubit_order = ops.QubitOrder.DEFAULT
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qubits = qubit_order.order_for(self.qubits)
        mat = np.ones(1, dtype=np.complex128)
        for qubit in qubits:
            oneq_state = self[qubit]
            state_vector = oneq_state.state_vector()
            mat = np.kron(mat, state_vector)
        return mat

    def projector(self, qubit_order: Optional['cirq.QubitOrder']=None) -> np.ndarray:
        """The projector associated with this state expressed as a matrix.

        This is |s⟩⟨s| where |s⟩ is this state.
        """
        from cirq import ops
        if qubit_order is None:
            qubit_order = ops.QubitOrder.DEFAULT
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qubits = qubit_order.order_for(self.qubits)
        mat = np.ones(1, dtype=np.complex128)
        for qubit in qubits:
            oneq_state = self[qubit]
            oneq_proj = oneq_state.projector()
            mat = np.kron(mat, oneq_proj)
        return mat