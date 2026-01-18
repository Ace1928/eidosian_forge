import numpy as np
import cirq
def test_via_gate_of_op():
    assert cirq.has_stabilizer_effect(YesOp())
    assert not cirq.has_stabilizer_effect(NoOp1())
    assert not cirq.has_stabilizer_effect(NoOp2())
    assert not cirq.has_stabilizer_effect(NoOp3())