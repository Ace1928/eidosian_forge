import pytest
import numpy as np
import cirq
from cirq_pasqal import ThreeDQubit, TwoDQubit
def test_triangular_2d():
    assert TwoDQubit.triangular_lattice(1) == [TwoDQubit(0.0, 0.0), TwoDQubit(0.5, 0.8660254037844386), TwoDQubit(1.0, 0.0), TwoDQubit(1.5, 0.8660254037844386)]
    assert TwoDQubit.triangular_lattice(1, x0=5.0, y0=6.1) == [TwoDQubit(5.0, 6.1), TwoDQubit(5.5, 6.966025403784438), TwoDQubit(6.0, 6.1), TwoDQubit(6.5, 6.966025403784438)]