import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
class UnsupportedGate(cirq.testing.TwoQubitGate):
    pass