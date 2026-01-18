import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
def test_teleportation_diagram():
    ali = cirq.NamedQubit('alice')
    car = cirq.NamedQubit('carrier')
    bob = cirq.NamedQubit('bob')
    circuit = cirq.Circuit(cirq.H(car), cirq.CNOT(car, bob), cirq.X(ali) ** 0.5, cirq.CNOT(ali, car), cirq.H(ali), [cirq.measure(ali), cirq.measure(car)], cirq.CNOT(car, bob), cirq.CZ(ali, bob))
    expected_diagram = '\n\\Qcircuit @R=1em @C=0.75em {\n \\\\\n &\\lstick{\\text{alice}}&   \\qw&\\gate{\\text{X}^{0.5}} \\qw&         \\qw    &\\control \\qw    &\\gate{\\text{H}} \\qw&\\meter   \\qw    &\\control \\qw    &\\qw\\\\\n &\\lstick{\\text{carrier}}& \\qw&\\gate{\\text{H}}       \\qw&\\control \\qw    &\\targ    \\qw\\qwx&\\meter          \\qw&\\control \\qw    &         \\qw\\qwx&\\qw\\\\\n &\\lstick{\\text{bob}}&     \\qw&                      \\qw&\\targ    \\qw\\qwx&         \\qw    &                \\qw&\\targ    \\qw\\qwx&\\control \\qw\\qwx&\\qw\\\\\n \\\\\n}'.strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram, qubit_order=cirq.QubitOrder.explicit([ali, car, bob]))