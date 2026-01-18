import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_numpy_ufunc():
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = np.sin(cirq.PauliString())
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        _ = np.exp(cirq.PauliString())
    x = np.exp(1j * np.pi * cirq.PauliString())
    assert x is not None