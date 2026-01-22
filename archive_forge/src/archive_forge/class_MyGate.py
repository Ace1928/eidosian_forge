import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class MyGate(cirq.Gate, cirq.InterchangeableQubitsGate):

    def __init__(self, num_qubits) -> None:
        self._num_qubits = num_qubits

    def num_qubits(self) -> int:
        return self._num_qubits

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        if index % 2 == 0:
            return index
        return 0