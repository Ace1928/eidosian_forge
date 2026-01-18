import pytest
import sympy
import numpy as np
import cirq
import cirq_pasqal
from cirq_pasqal import PasqalDevice, PasqalVirtualDevice
from cirq_pasqal import TwoDQubit, ThreeDQubit
def test_validate_moment():
    d = square_virtual_device(control_r=1.0, num_qubits=2)
    m1 = cirq.Moment([cirq.Z.on(TwoDQubit(0, 0)), cirq.X.on(TwoDQubit(1, 1))])
    m2 = cirq.Moment([cirq.Z.on(TwoDQubit(0, 0))])
    m3 = cirq.Moment([cirq.measure(TwoDQubit(0, 0)), cirq.measure(TwoDQubit(1, 1))])
    with pytest.raises(ValueError, match='Cannot do simultaneous gates'):
        d.validate_moment(m1)
    d.validate_moment(m2)
    d.validate_moment(m3)