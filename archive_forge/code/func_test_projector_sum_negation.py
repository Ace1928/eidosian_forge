import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_negation():
    q0 = cirq.NamedQubit('q0')
    zero_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    negation = -zero_projector_sum.copy()
    np.testing.assert_allclose(negation.matrix().toarray(), [[-1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(zero_projector_sum.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])