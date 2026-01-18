from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_simplifies_sqrt_iswap_inv():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(use_sqrt_iswap_inv=True, before=cirq.Circuit([cirq.Moment([cirq.SQRT_ISWAP(a, b)]), cirq.Moment([cirq.SQRT_ISWAP(a, b)]), cirq.Moment([cirq.SQRT_ISWAP(a, b)]), cirq.Moment([cirq.SQRT_ISWAP(a, b)]), cirq.Moment([cirq.SQRT_ISWAP(a, b)]), cirq.Moment([cirq.SQRT_ISWAP_INV(a, b)]), cirq.Moment([cirq.SQRT_ISWAP(a, b)]), cirq.Moment([cirq.SQRT_ISWAP(a, b)]), cirq.Moment([cirq.SQRT_ISWAP(a, b)])]), expected=cirq.Circuit([cirq.Moment([cirq.SQRT_ISWAP_INV(a, b)])]))