import numpy as np
import pytest
import cirq
from cirq.devices.noise_utils import (
def test_op_identifier():
    op_id = OpIdentifier(cirq.XPowGate)
    assert cirq.X(cirq.LineQubit(1)) in op_id
    assert cirq.Rx(rads=1) in op_id