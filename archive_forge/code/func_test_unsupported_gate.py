import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_unsupported_gate():

    class UnknownGate(cirq.testing.TwoQubitGate):
        pass
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(UnknownGate()(q0, q1))
    with pytest.raises(ValueError, match='Unable to convert'):
        cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset(), ignore_failures=False)