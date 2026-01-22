import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
@value.value_equality(unhashable=True)
class LinearPermutationGate(PermutationGate):
    """A permutation gate that decomposes a given permutation using a linear
    sorting network."""

    def __init__(self, num_qubits: int, permutation: Dict[int, int], swap_gate: 'cirq.Gate'=ops.SWAP) -> None:
        """Initializes a linear permutation gate.

        Args:
            num_qubits: The number of qubits to permute.
            permutation: The permutation effected by the gate.
            swap_gate: The swap gate used in decompositions.
        """
        super().__init__(num_qubits, swap_gate)
        PermutationGate.validate_permutation(permutation, num_qubits)
        self._permutation = permutation

    def permutation(self) -> Dict[int, int]:
        return self._permutation

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        swap_gate = SwapPermutationGate(self.swap_gate)
        n_qubits = len(qubits)
        mapping = {i: self._permutation.get(i, i) for i in range(n_qubits)}
        for layer_index in range(n_qubits):
            for i in range(layer_index % 2, n_qubits - 1, 2):
                if mapping[i] > mapping[i + 1]:
                    yield swap_gate(*qubits[i:i + 2])
                    mapping[i], mapping[i + 1] = (mapping[i + 1], mapping[i])

    def __repr__(self) -> str:
        return f'cirq.contrib.acquaintance.LinearPermutationGate({self.num_qubits()!r}, {self._permutation!r}, {self.swap_gate!r})'

    def _value_equality_values_(self) -> Any:
        return (tuple(sorted(((i, j) for i, j in self._permutation.items() if i != j))), self.swap_gate)

    def __bool__(self) -> bool:
        return bool(_canonicalize_permutation(self._permutation))

    def __pow__(self, exponent):
        if exponent == 1:
            return self
        if exponent == -1:
            return LinearPermutationGate(self._num_qubits, {v: k for k, v in self._permutation.items()}, self.swap_gate)
        return NotImplemented