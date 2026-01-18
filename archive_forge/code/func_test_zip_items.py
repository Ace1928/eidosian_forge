import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('map1,map2,out', (lambda q0, q1, q2: (({}, {}, {}), ({q0: cirq.X}, {q0: cirq.Y}, {q0: (cirq.X, cirq.Y)}), ({q0: cirq.X}, {q1: cirq.X}, {}), ({q0: cirq.Y, q1: cirq.Z}, {q1: cirq.Y, q2: cirq.X}, {q1: (cirq.Z, cirq.Y)}), ({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}, {}, {}), ({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}, {q0: cirq.Y, q1: cirq.Z}, {q0: (cirq.X, cirq.Y), q1: (cirq.Y, cirq.Z)})))(*_make_qubits(3)))
def test_zip_items(map1, map2, out):
    ps1 = cirq.PauliString(map1)
    ps2 = cirq.PauliString(map2)
    out_actual = tuple(ps1.zip_items(ps2))
    assert len(out_actual) == len(out)
    assert dict(out_actual) == out