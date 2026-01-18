import pytest
import sympy
import numpy as np
import cirq
import cirq_pasqal
from cirq_pasqal import PasqalDevice, PasqalVirtualDevice
from cirq_pasqal import TwoDQubit, ThreeDQubit
def test_validate_circuit():
    d = generic_device(2)
    circuit1 = cirq.Circuit()
    circuit1.append(cirq.X(cirq.NamedQubit('q1')))
    circuit1.append(cirq.measure(cirq.NamedQubit('q1')))
    d.validate_circuit(circuit1)
    circuit1.append(cirq.CX(cirq.NamedQubit('q1'), cirq.NamedQubit('q0')))
    with pytest.raises(ValueError, match='Non-empty moment after measurement'):
        d.validate_circuit(circuit1)