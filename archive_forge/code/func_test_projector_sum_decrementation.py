import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_decrementation():
    q0 = cirq.NamedQubit('q0')
    zero_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    one_projector_string = cirq.ProjectorString({q0: 1})
    one_projector_sum = cirq.ProjectorSum.from_projector_strings(one_projector_string)
    decrementation = zero_projector_sum.copy()
    decrementation -= one_projector_sum
    np.testing.assert_allclose(decrementation.matrix().toarray(), [[1.0, 0.0], [0.0, -1.0]])
    decrementation = zero_projector_sum.copy()
    decrementation -= one_projector_string
    np.testing.assert_allclose(decrementation.matrix().toarray(), [[1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(zero_projector_sum.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(one_projector_sum.matrix().toarray(), [[0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError):
        zero_projector_sum -= 0.12345