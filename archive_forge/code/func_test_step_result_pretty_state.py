import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_step_result_pretty_state():
    step_result = BasicStateVector()
    assert step_result.dirac_notation() == '|01⟩'