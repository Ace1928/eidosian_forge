import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class Inconsistent(cirq.testing.SingleQubitGate):

    def __repr__(self):
        return 'Inconsistent'

    def on(self, *qubits):
        return cirq.GateOperation(Inconsistent(), qubits)