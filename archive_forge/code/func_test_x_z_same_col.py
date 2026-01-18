import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_x_z_same_col():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.X(a), cirq.Z(b))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["X","Z"]]}\n    ', escape_url=False)
    assert_links_to(circuit, 'http://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22X%22%2C%22Z%22%5D%5D%7D')