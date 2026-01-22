import pytest
import numpy as np
import sympy
import cirq
class GateDecomposesToDefaultGateset(cirq.Gate):

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        return [GoodGateDecompose().on(qubits[0]), BadGateDecompose().on(qubits[1])]