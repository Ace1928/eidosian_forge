import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class GateRMul(cirq.Gate):

    def num_qubits(self) -> int:
        return 1

    def _rmul_with_qubits(self, qubits, other):
        if other == 2:
            return 3
        if isinstance(other, cirq.Operation) and isinstance(other.gate, GateRMul):
            return 4
        raise NotImplementedError()