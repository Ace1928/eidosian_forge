import pytest
import numpy as np
import cirq
from cirq_pasqal import ThreeDQubit, TwoDQubit
def test_pasqal_qubit_ordering_3d():
    assert ThreeDQubit(0, 0, 1) >= ThreeDQubit(1, 0, 0)
    assert ThreeDQubit(0, 0, 1) >= ThreeDQubit(0, 1, 0)
    assert ThreeDQubit(0, 1, 0) >= ThreeDQubit(1, 0, 0)
    for i in range(8):
        v = [int(x) for x in bin(i)[2:].zfill(3)]
        assert ThreeDQubit(0, 0, 0) <= ThreeDQubit(v[0], v[1], v[2])
        assert ThreeDQubit(1, 1, 1) >= ThreeDQubit(v[0], v[1], v[2])
        if i >= 1:
            assert ThreeDQubit(0, 0, 0) < ThreeDQubit(v[0], v[1], v[2])
        if i < 7:
            assert ThreeDQubit(1, 1, 1) > ThreeDQubit(v[0], v[1], v[2])