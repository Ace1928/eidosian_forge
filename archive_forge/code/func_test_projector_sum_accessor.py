import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_accessor():
    q0 = cirq.NamedQubit('q0')
    projector_string_1 = cirq.ProjectorString({q0: 0}, 0.2016)
    projector_string_2 = cirq.ProjectorString({q0: 1}, 0.0913)
    projector_sum = cirq.ProjectorSum.from_projector_strings([projector_string_1, projector_string_2])
    assert len(projector_sum) == 2
    expanded_projector_strings = list(projector_sum)
    assert expanded_projector_strings == [projector_string_1, projector_string_2]