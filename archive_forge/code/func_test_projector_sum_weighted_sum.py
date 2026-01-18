import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_weighted_sum():
    q0 = cirq.NamedQubit('q0')
    zero_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    one_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 1}))
    weighted_sum = 0.6 * zero_projector_sum + 0.4 * one_projector_sum
    np.testing.assert_allclose(weighted_sum.matrix().toarray(), [[0.6, 0.0], [0.0, 0.4]])
    np.testing.assert_allclose(zero_projector_sum.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(one_projector_sum.matrix().toarray(), [[0.0, 0.0], [0.0, 1.0]])