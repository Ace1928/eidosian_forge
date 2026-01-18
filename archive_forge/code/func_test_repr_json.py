import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
@pytest.mark.parametrize('gateset', [cirq_google.SycamoreTargetGateset(), cirq_google.SycamoreTargetGateset(tabulation=cirq.two_qubit_gate_product_tabulation(cirq.unitary(cirq_google.SYC), 0.1, random_state=cirq.value.parse_random_state(11)))])
def test_repr_json(gateset):
    assert eval(repr(gateset)) == gateset
    assert cirq.read_json(json_text=cirq.to_json(gateset)) == gateset