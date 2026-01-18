import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_expectations_matrix():
    q0 = cirq.NamedQubit('q0')
    zero_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}, coefficient=0.2016))
    one_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 1}, coefficient=0.0913))
    proj_sum = 0.6 * zero_projector_sum + 0.4 * one_projector_sum
    np.testing.assert_allclose(proj_sum.matrix().toarray(), 0.6 * zero_projector_sum.matrix().toarray() + 0.4 * one_projector_sum.matrix().toarray())