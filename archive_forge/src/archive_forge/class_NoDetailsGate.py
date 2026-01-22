import numpy as np
import pytest
import sympy
import cirq
class NoDetailsGate(cirq.Gate):

    def num_qubits(self) -> int:
        return 1