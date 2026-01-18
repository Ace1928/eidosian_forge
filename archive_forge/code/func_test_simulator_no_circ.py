import pytest
import cirq
from cirq_aqt import AQTSimulator
from cirq_aqt.aqt_device import get_aqt_device
from cirq_aqt.aqt_device import AQTNoiseModel
def test_simulator_no_circ():
    with pytest.raises(RuntimeError):
        sim = AQTSimulator(num_qubits=1)
        sim.simulate_samples(1)