from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_sqrt_iswap_gateset_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.SqrtIswapTargetGateset(), cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=False))
    eq.add_equality_group(cirq.SqrtIswapTargetGateset(atol=1e-06, required_sqrt_iswap_count=0, use_sqrt_iswap_inv=True))
    eq.add_equality_group(cirq.SqrtIswapTargetGateset(atol=1e-06, required_sqrt_iswap_count=3, use_sqrt_iswap_inv=True))
    eq.add_equality_group(cirq.SqrtIswapTargetGateset(additional_gates=[cirq.XPowGate]))