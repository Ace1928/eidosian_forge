import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_x_cnot_split_cols():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = cirq.Circuit(cirq.CNOT(a, b), cirq.X(c))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["•","X"],[1,1,"X"]]}\n    ', escape_url=False)