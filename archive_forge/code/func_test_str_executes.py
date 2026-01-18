import re
import numpy as np
import pytest
import sympy
import cirq
def test_str_executes():
    assert '1' in str(cirq.MatrixGate(np.eye(2)))
    assert '0' in str(cirq.MatrixGate(np.eye(4)))