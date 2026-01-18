import numpy as np
import pytest
import sympy
import cirq
def test_zz_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.ZZ(a, b), cirq.ZZ(a, b) ** 3, cirq.ZZ(a, b) ** 0.5)
    cirq.testing.assert_has_diagram(circuit, '\na: ───ZZ───ZZ───ZZ───────\n      │    │    │\nb: ───ZZ───ZZ───ZZ^0.5───\n')