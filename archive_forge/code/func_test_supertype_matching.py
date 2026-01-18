import cirq
from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG, OpIdentifier
def test_supertype_matching():
    q0 = cirq.LineQubit(0)
    op_id0 = OpIdentifier(cirq.Gate, q0)
    op_id1 = OpIdentifier(cirq.XPowGate, q0)
    model = InsertionNoiseModel({op_id0: cirq.T(q0), op_id1: cirq.S(q0)}, require_physical_tag=False)
    moment_0 = cirq.Moment(cirq.Rx(rads=1).on(q0))
    assert model.noisy_moment(moment_0, system_qubits=[q0]) == [moment_0, cirq.Moment(cirq.S(q0))]
    moment_1 = cirq.Moment(cirq.Y(q0))
    assert model.noisy_moment(moment_1, system_qubits=[q0]) == [moment_1, cirq.Moment(cirq.T(q0))]
    cirq.testing.assert_equivalent_repr(model)