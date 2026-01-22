import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class GateMul(cirq.Gate):

    def num_qubits(self) -> int:
        return 1

    def _mul_with_qubits(self, qubits, other):
        if other == 2:
            return 5
        if isinstance(other, cirq.Operation) and isinstance(other.gate, GateMul):
            return 6
        raise NotImplementedError()