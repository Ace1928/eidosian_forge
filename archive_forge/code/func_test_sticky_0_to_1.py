import itertools
import random
from typing import Any
import numpy as np
import pytest
import sympy
import cirq
from cirq.transformers.analytical_decompositions.two_qubit_to_fsim import (
def test_sticky_0_to_1():
    assert _sticky_0_to_1(-1, atol=1e-08) is None
    assert _sticky_0_to_1(-1e-06, atol=1e-08) is None
    assert _sticky_0_to_1(-1e-10, atol=1e-08) == 0
    assert _sticky_0_to_1(0, atol=1e-08) == 0
    assert _sticky_0_to_1(1e-10, atol=1e-08) == 1e-10
    assert _sticky_0_to_1(1e-06, atol=1e-08) == 1e-06
    assert _sticky_0_to_1(0.5, atol=1e-08) == 0.5
    assert _sticky_0_to_1(1 - 1e-06, atol=1e-08) == 1 - 1e-06
    assert _sticky_0_to_1(1 - 1e-10, atol=1e-08) == 1 - 1e-10
    assert _sticky_0_to_1(1, atol=1e-08) == 1
    assert _sticky_0_to_1(1 + 1e-10, atol=1e-08) == 1
    assert _sticky_0_to_1(1 + 1e-06, atol=1e-08) is None
    assert _sticky_0_to_1(2, atol=1e-08) is None
    assert _sticky_0_to_1(-0.1, atol=0.5) == 0