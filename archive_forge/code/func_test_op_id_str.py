import numpy as np
import pytest
import cirq
from cirq.devices.noise_utils import (
def test_op_id_str():
    op_id = OpIdentifier(cirq.XPowGate, cirq.LineQubit(0))
    assert str(op_id) == "<class 'cirq.ops.common_gates.XPowGate'>(cirq.LineQubit(0),)"
    assert repr(op_id) == 'cirq.devices.noise_utils.OpIdentifier(cirq.ops.common_gates.XPowGate, cirq.LineQubit(0))'