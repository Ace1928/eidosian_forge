import numpy as np
import pytest
import sympy
import cirq
def test_xx_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.XX(a, b), cirq.XX(a, b) ** 3, cirq.XX(a, b) ** 0.5)
    cirq.testing.assert_has_diagram(circuit, '\na: ───XX───XX───XX───────\n      │    │    │\nb: ───XX───XX───XX^0.5───\n')