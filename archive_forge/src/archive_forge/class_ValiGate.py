from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class ValiGate(cirq.Gate):

    def _num_qubits_(self):
        return 2

    def validate_args(self, qubits):
        if len(qubits) == 1:
            return
        super().validate_args(qubits)

    def _has_mixture_(self):
        return True