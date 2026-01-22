import json
import time
import uuid
from typing import cast, Dict, List, Sequence, Tuple, Union
import numpy as np
from requests import put
import cirq
from cirq_aqt.aqt_device import AQTSimulator, get_op_string
class AQTSamplerLocalSimulator(AQTSampler):
    """cirq.Sampler using the AQT simulator on the local machine.

    Can be used as a replacement for the AQTSampler
    When the attribute simulate_ideal is set to True,
    an ideal circuit is sampled
    If not, the error model defined in aqt_simulator_test.py is used
    Example for running the ideal sampler:

    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal=True
    """

    def __init__(self, remote_host: str='', access_token: str='', simulate_ideal: bool=False):
        """Args:
        remote_host: Remote host is not used by the local simulator.
        access_token: Access token is not used by the local simulator.
        simulate_ideal: Boolean that determines whether a noisy or
                        an ideal simulation is performed.
        """
        self.remote_host = remote_host
        self.access_token = access_token
        self.simulate_ideal = simulate_ideal

    def _send_json(self, *, json_str: str, id_str: Union[str, uuid.UUID], repetitions: int=1, num_qubits: int=1) -> np.ndarray:
        """Replaces the remote host with a local simulator

        Args:
            json_str: Json representation of the circuit.
            id_str: Unique id of the datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.

        Returns:
            Measurement results as an ndarray of booleans.
        """
        sim = AQTSimulator(num_qubits=num_qubits, simulate_ideal=self.simulate_ideal)
        sim.generate_circuit_from_list(json_str)
        data = sim.simulate_samples(repetitions)
        return data.measurements['m']