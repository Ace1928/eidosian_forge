import numpy as np
import pytest
import cirq
import cirq_google as cg
def test_syc_circuit_diagram():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cg.SYC(a, b))
    cirq.testing.assert_has_diagram(circuit, '\n0: ───SYC───\n      │\n1: ───SYC───\n')