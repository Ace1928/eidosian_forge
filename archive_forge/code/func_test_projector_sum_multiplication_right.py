import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_multiplication_right():
    q0 = cirq.NamedQubit('q0')
    zero_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    multiplication_float = zero_projector_sum * 2.0
    np.testing.assert_allclose(multiplication_float.matrix().toarray(), [[2.0, 0.0], [0.0, 0.0]])
    multiplication_int = zero_projector_sum * 2
    np.testing.assert_allclose(multiplication_int.matrix().toarray(), [[2.0, 0.0], [0.0, 0.0]])
    multiplication_complex = zero_projector_sum * 2j
    np.testing.assert_allclose(multiplication_complex.matrix().toarray(), [[2j, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(zero_projector_sum.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(TypeError):
        _ = zero_projector_sum * 'not_the_correct_type'