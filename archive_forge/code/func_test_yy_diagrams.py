import numpy as np
import pytest
import sympy
import cirq
def test_yy_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.YY(a, b), cirq.YY(a, b) ** 3, cirq.YY(a, b) ** 0.5)
    cirq.testing.assert_has_diagram(circuit, '\na: ───YY───YY───YY───────\n      │    │    │\nb: ───YY───YY───YY^0.5───\n')