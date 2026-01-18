import numpy as np
import pytest
import cirq
from cirq.devices.noise_utils import (
def test_op_id_instance():
    q0 = cirq.LineQubit.range(1)[0]
    gate = cirq.SingleQubitCliffordGate.from_xz_map((cirq.X, False), (cirq.Z, False))
    op_id = OpIdentifier(gate, q0)
    cirq.testing.assert_equivalent_repr(op_id)