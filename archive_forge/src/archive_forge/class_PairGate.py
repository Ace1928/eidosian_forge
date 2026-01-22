import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@cirq.value_equality
class PairGate(cirq.Gate, cirq.InterchangeableQubitsGate):
    """Interchangeable subsets."""

    def __init__(self, num_qubits):
        self._num_qubits = num_qubits

    def num_qubits(self) -> int:
        return self._num_qubits

    def qubit_index_to_equivalence_group_key(self, index: int):
        return index // 2

    def _value_equality_values_(self):
        return (self.num_qubits(),)