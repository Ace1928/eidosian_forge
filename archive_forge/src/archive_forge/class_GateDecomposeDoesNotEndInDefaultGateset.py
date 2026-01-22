import pytest
import numpy as np
import sympy
import cirq
class GateDecomposeDoesNotEndInDefaultGateset(cirq.Gate):

    def _num_qubits_(self):
        return 4

    def _decompose_(self, qubits):
        yield GateDecomposeNotImplemented().on_each(*qubits)