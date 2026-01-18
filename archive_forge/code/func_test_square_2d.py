import pytest
import numpy as np
import cirq
from cirq_pasqal import ThreeDQubit, TwoDQubit
def test_square_2d():
    assert TwoDQubit.square(2, x0=1, y0=1) == [TwoDQubit(1, 1), TwoDQubit(2, 1), TwoDQubit(1, 2), TwoDQubit(2, 2)]
    assert TwoDQubit.square(2) == [TwoDQubit(0, 0), TwoDQubit(1, 0), TwoDQubit(0, 1), TwoDQubit(1, 1)]