from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class Gate2(cirq.Gate):

    def _decompose_(self, qubits):
        return cirq.S.on(qubits[0])

    def num_qubits(self) -> int:
        return 1

    def _circuit_diagram_info_(self, args):
        return 's!'