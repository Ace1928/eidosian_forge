from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('theta, phi', itertools.product((-2.4 * np.pi, -np.pi / 11, np.pi / 9, np.pi / 2, 1.4 * np.pi), (-1.4 * np.pi, -np.pi / 9, np.pi / 11, np.pi / 2, 2.4 * np.pi)))
def test_valid_cphase_exponents(theta, phi):
    fsim_gate = cirq.FSimGate(theta=theta, phi=phi)
    valid_exponent_intervals = cirq.compute_cphase_exponents_for_fsim_decomposition(fsim_gate)
    assert valid_exponent_intervals
    for min_exponent, max_exponent in valid_exponent_intervals:
        margin = 1e-08
        min_exponent += margin
        max_exponent -= margin
        assert min_exponent < max_exponent
        for exponent in np.linspace(min_exponent, max_exponent, 3):
            for d in (-2, 0, 4):
                cphase_gate = cirq.CZPowGate(exponent=exponent + d)
                assert_decomposition_valid(cphase_gate, fsim_gate=fsim_gate)
                cphase_gate = cirq.CZPowGate(exponent=-exponent + d)
                assert_decomposition_valid(cphase_gate, fsim_gate=fsim_gate)