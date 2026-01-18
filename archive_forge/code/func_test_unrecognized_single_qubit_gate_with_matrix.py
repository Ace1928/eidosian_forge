import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_unrecognized_single_qubit_gate_with_matrix():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.PhasedXPowGate(phase_exponent=0).on(a) ** 0.2731)
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[[{\n            "id":"?",\n            "matrix":"{\n                {0.826988+0.378258i, 0.173012-0.378258i},\n                {0.173012-0.378258i, 0.826988+0.378258i}\n            }"}]]}\n    ', escape_url=False)