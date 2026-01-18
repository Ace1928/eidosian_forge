import numpy as np
import pytest
import sympy
from scipy import linalg
import cirq
def test_trace_distance_over_range_of_exponents():
    for exp in np.linspace(0, 4, 20):
        cirq.testing.assert_has_consistent_trace_distance_bound(cirq.SWAP ** exp)
        cirq.testing.assert_has_consistent_trace_distance_bound(cirq.ISWAP ** exp)