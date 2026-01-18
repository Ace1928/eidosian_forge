from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_works_with_tags():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.SQRT_ISWAP(a, b).with_tags('mytag1')]), cirq.Moment([cirq.SQRT_ISWAP(a, b).with_tags('mytag2')]), cirq.Moment([cirq.SQRT_ISWAP_INV(a, b).with_tags('mytag3')])]), expected=cirq.Circuit([cirq.Moment([cirq.SQRT_ISWAP(a, b)])]))