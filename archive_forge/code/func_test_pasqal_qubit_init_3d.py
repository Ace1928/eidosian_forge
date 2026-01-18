import pytest
import numpy as np
import cirq
from cirq_pasqal import ThreeDQubit, TwoDQubit
def test_pasqal_qubit_init_3d():
    q = ThreeDQubit(3, 4, 5)
    assert q.x == 3
    assert q.y == 4
    assert q.z == 5