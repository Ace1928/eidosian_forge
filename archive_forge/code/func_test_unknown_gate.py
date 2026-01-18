import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_unknown_gate():

    class UnknownGate(cirq.testing.SingleQubitGate):
        pass
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(UnknownGate()(a))
    with pytest.raises(TypeError):
        _ = circuit_to_quirk_url(circuit)
    with pytest.raises(TypeError):
        _ = circuit_to_quirk_url(circuit, escape_url=False)
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["UNKNOWN"]]}\n    ', prefer_unknown_gate_to_failure=True, escape_url=False)