import abc
from typing import Collection, Tuple, TYPE_CHECKING, Any, Dict, Iterator, Optional, Sequence, Union
import itertools
from cirq import protocols, value, _compat
class SumOfProducts(AbstractControlValues):
    """Represents control values as AND (product) clauses, each of which applies to all N qubits.

    `SumOfProducts` representation describes the control values as a union
    of n-bit tuples, where each n-bit tuple represents an allowed assignment
    of bits for which the control should be activated. This expanded
    representation allows us to create control values combinations which
    cannot be factored as a `ProductOfSums` representation.

    For example:

    1) `(|00><00| + |11><11|) X + (|01><01| + |10><10|) I` represents an
        operator which flips the third qubit if the first two qubits
        are `00` or `11`, and does nothing otherwise.
        This can be constructed as
        >>> xor_control_values = cirq.SumOfProducts(((0, 0), (1, 1)))
        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> xor_cop = cirq.X(q2).controlled_by(q0, q1, control_values=xor_control_values)

    2) `(|00><00| + |01><01| + |10><10|) X + (|11><11|) I` represents an
        operators which flips the third qubit if the `nand` of first two
        qubits is `1` (i.e. first two qubits are either `00`, `01` or `10`),
        and does nothing otherwise. This can be constructed as:

        >>> nand_control_values = cirq.SumOfProducts(((0, 0), (0, 1), (1, 0)))
        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> nand_cop = cirq.X(q2).controlled_by(q0, q1, control_values=nand_control_values)
    """

    def __init__(self, data: Collection[Sequence[int]], *, name: Optional[str]=None):
        self._conjunctions: Tuple[Tuple[int, ...], ...] = tuple(sorted(set((tuple(cv) for cv in data))))
        self._name = name
        if not len(self._conjunctions):
            raise ValueError("SumOfProducts can't be empty.")
        num_qubits = len(self._conjunctions[0])
        if not all((len(p) == num_qubits for p in self._conjunctions)):
            raise ValueError(f'Each term of {self._conjunctions} should be of length {num_qubits}.')

    @_compat.cached_property
    def is_trivial(self) -> bool:
        return self._conjunctions == ((1,) * self._num_qubits_(),)

    def expand(self) -> 'SumOfProducts':
        return self

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        """Returns the combinations tracked by the object."""
        return iter(self._conjunctions)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        """Returns a string representation to be used in circuit diagrams."""
        if self._name is not None:
            wire_symbols = ['@'] * self._num_qubits_()
            wire_symbols[-1] = f'@({self._name})'
            return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols)
        if len(self._conjunctions) == 1:
            return protocols.CircuitDiagramInfo(wire_symbols=['@' if x == 1 else f'({x})' for x in self._conjunctions[0]])
        wire_symbols = [''] * self._num_qubits_()
        for term in self._conjunctions:
            for q_i, q_val in enumerate(term):
                wire_symbols[q_i] = wire_symbols[q_i] + str(q_val)
        wire_symbols = [f'@({s})' for s in wire_symbols]
        return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __repr__(self) -> str:
        name = '' if self._name is None else f', name="{self._name}"'
        return f'cirq.SumOfProducts({self._conjunctions!s} {name})'

    def __str__(self) -> str:
        suffix = self._name if self._name is not None else '_'.join((''.join((str(v) for v in t)) for t in self._conjunctions))
        return f'C_{suffix}'

    def _num_qubits_(self) -> int:
        return len(self._conjunctions[0])

    def validate(self, qid_shapes: Sequence[int]) -> None:
        if len(qid_shapes) != self._num_qubits_():
            raise ValueError(f'Length of qid_shapes: {qid_shapes} should be equal to self._num_qubits_(): {self._num_qubits_()}')
        for product in self._conjunctions:
            for q_i, q_val in enumerate(product):
                if not 0 <= q_val < qid_shapes[q_i]:
                    raise ValueError(f'Control value <{q_val}> in combination {product} is outside of range [0, {qid_shapes[q_i]}) for control qubit number <{q_i}>.')

    def _json_dict_(self) -> Dict[str, Any]:
        return {'data': self._conjunctions, 'name': self._name}