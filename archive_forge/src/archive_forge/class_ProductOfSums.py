import abc
from typing import Collection, Tuple, TYPE_CHECKING, Any, Dict, Iterator, Optional, Sequence, Union
import itertools
from cirq import protocols, value, _compat
class ProductOfSums(AbstractControlValues):
    """Represents control values as N OR (sum) clauses, each of which applies to one qubit."""

    def __init__(self, data: Sequence[Union[int, Collection[int]]]):
        self._qubit_sums: Tuple[Tuple[int, ...], ...] = tuple(((cv,) if isinstance(cv, int) else tuple(sorted(set(cv))) for cv in data))

    @_compat.cached_property
    def is_trivial(self) -> bool:
        return self._qubit_sums == ((1,),) * self._num_qubits_()

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        return iter(self._qubit_sums)

    def expand(self) -> 'SumOfProducts':
        return SumOfProducts(tuple(itertools.product(*self._qubit_sums)))

    def __repr__(self) -> str:
        return f'cirq.ProductOfSums({str(self._qubit_sums)})'

    def _num_qubits_(self) -> int:
        return len(self._qubit_sums)

    def __getitem__(self, key: Union[int, slice]) -> Union['ProductOfSums', Tuple[int, ...]]:
        if isinstance(key, slice):
            return ProductOfSums(self._qubit_sums[key])
        return self._qubit_sums[key]

    def validate(self, qid_shapes: Sequence[int]) -> None:
        for i, (vals, shape) in enumerate(zip(self._qubit_sums, qid_shapes)):
            if not all((0 <= v < shape for v in vals)):
                message = f'Control values <{vals!r}> outside of range for control qubit number <{i}>.'
                raise ValueError(message)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        """Returns a string representation to be used in circuit diagrams."""

        def get_symbol(vals):
            return '@' if tuple(vals) == (1,) else f'({','.join(map(str, vals))})'
        return protocols.CircuitDiagramInfo(wire_symbols=[get_symbol(t) for t in self._qubit_sums])

    def __str__(self) -> str:
        if self.is_trivial:
            return 'C' * self._num_qubits_()

        def get_prefix(control_vals):
            control_vals_str = ''.join(map(str, sorted(control_vals)))
            return f'C{control_vals_str}'
        return ''.join((get_prefix(t) for t in self._qubit_sums))

    def _json_dict_(self) -> Dict[str, Any]:
        return {'data': self._qubit_sums}

    def __and__(self, other: AbstractControlValues) -> AbstractControlValues:
        if isinstance(other, ProductOfSums):
            return ProductOfSums(self._qubit_sums + other._qubit_sums)
        return super().__and__(other)

    def __or__(self, other: AbstractControlValues) -> AbstractControlValues:
        if protocols.num_qubits(self) != protocols.num_qubits(other):
            raise ValueError(f'Control values {self} and {other} must act on equal number of qubits')
        if isinstance(other, ProductOfSums):
            return ProductOfSums(tuple((x + y for x, y in zip(self._qubit_sums, other._qubit_sums))))
        return super().__or__(other)