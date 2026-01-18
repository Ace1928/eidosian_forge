import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('psum, exp', ((cirq.Z(q0), np.pi / 2), (2 * cirq.X(q0) + 3 * cirq.Y(q2), 1), (cirq.X(q0) * cirq.Y(q1) + cirq.Y(q1) * cirq.Z(q3), np.pi)))
def test_with_parameters_resolved_by(psum, exp):
    psum_exp = cirq.PauliSumExponential(psum, sympy.Symbol('theta'))
    resolver = cirq.ParamResolver({'theta': exp})
    actual = cirq.resolve_parameters(psum_exp, resolver)
    expected = cirq.PauliSumExponential(psum, exp)
    assert actual == expected