from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
@pytest.mark.parametrize('gateset', [cirq.SqrtIswapTargetGateset(), cirq.SqrtIswapTargetGateset(atol=1e-06, required_sqrt_iswap_count=2, use_sqrt_iswap_inv=True, additional_gates=[cirq.CZ, cirq.XPowGate, cirq.YPowGate, cirq.GateFamily(cirq.ZPowGate, tags_to_accept=['test_tag'])]), cirq.SqrtIswapTargetGateset(additional_gates=())])
def test_sqrt_iswap_gateset_repr(gateset):
    cirq.testing.assert_equivalent_repr(gateset)