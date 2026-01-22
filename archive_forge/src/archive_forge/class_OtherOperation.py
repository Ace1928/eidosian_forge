from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
class OtherOperation(cirq.Operation):

    def __init__(self, qubits: Sequence[cirq.Qid]) -> None:
        self._qubits = tuple(qubits)

    @property
    def qubits(self) -> Tuple[cirq.Qid, ...]:
        return self._qubits

    def with_qubits(self, *new_qubits: cirq.Qid) -> 'OtherOperation':
        return type(self)(self._qubits)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.qubits == other.qubits