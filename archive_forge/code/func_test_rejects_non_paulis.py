import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_rejects_non_paulis():
    q = cirq.NamedQubit('q')
    with pytest.raises(TypeError):
        _ = cirq.PauliString({q: cirq.S})