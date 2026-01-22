import warnings
import collections
import json
import os
import re
from typing import List, Iterable
from qiskit import circuit
from qiskit.providers.models import BackendProperties, BackendConfiguration, PulseDefaults
from qiskit.providers import BackendV2, BackendV1
from qiskit import pulse
from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.providers import basic_provider
from qiskit.transpiler import Target
from qiskit.providers.backend_compat import convert_to_target
from .utils.json_decoder import (
class FakeBackend(BackendV1):
    """This is a dummy backend just for testing purposes."""

    def __init__(self, configuration, time_alive=10):
        """FakeBackend initializer.

        Args:
            configuration (BackendConfiguration): backend configuration
            time_alive (int): time to wait before returning result
        """
        super().__init__(configuration)
        self.time_alive = time_alive
        self._credentials = _Credentials()
        self.sim = None

    def _setup_sim(self):
        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel
            self.sim = AerSimulator()
            if self.properties():
                noise_model = NoiseModel.from_backend(self)
                self.sim.set_options(noise_model=noise_model)
                self.set_options(noise_model=noise_model)
        else:
            self.sim = basic_provider.BasicSimulator()

    def properties(self):
        """Return backend properties"""
        coupling_map = self.configuration().coupling_map
        if coupling_map is None:
            return None
        unique_qubits = list(set().union(*coupling_map))
        properties = {'backend_name': self.name(), 'backend_version': self.configuration().backend_version, 'last_update_date': '2000-01-01 00:00:00Z', 'qubits': [[{'date': '2000-01-01 00:00:00Z', 'name': 'T1', 'unit': 'µs', 'value': 0.0}, {'date': '2000-01-01 00:00:00Z', 'name': 'T2', 'unit': 'µs', 'value': 0.0}, {'date': '2000-01-01 00:00:00Z', 'name': 'frequency', 'unit': 'GHz', 'value': 0.0}, {'date': '2000-01-01 00:00:00Z', 'name': 'readout_error', 'unit': '', 'value': 0.0}, {'date': '2000-01-01 00:00:00Z', 'name': 'operational', 'unit': '', 'value': 1}] for _ in range(len(unique_qubits))], 'gates': [{'gate': 'cx', 'name': 'CX' + str(pair[0]) + '_' + str(pair[1]), 'parameters': [{'date': '2000-01-01 00:00:00Z', 'name': 'gate_error', 'unit': '', 'value': 0.0}], 'qubits': [pair[0], pair[1]]} for pair in coupling_map], 'general': []}
        return BackendProperties.from_dict(properties)

    @classmethod
    def _default_options(cls):
        if _optionals.HAS_AER:
            from qiskit_aer import QasmSimulator
            return QasmSimulator._default_options()
        else:
            return basic_provider.BasicSimulator._default_options()

    def run(self, run_input, **kwargs):
        """Main job in simulator"""
        circuits = run_input
        pulse_job = None
        if isinstance(circuits, (pulse.Schedule, pulse.ScheduleBlock)):
            pulse_job = True
        elif isinstance(circuits, circuit.QuantumCircuit):
            pulse_job = False
        elif isinstance(circuits, list):
            if circuits:
                if all((isinstance(x, (pulse.Schedule, pulse.ScheduleBlock)) for x in circuits)):
                    pulse_job = True
                elif all((isinstance(x, circuit.QuantumCircuit) for x in circuits)):
                    pulse_job = False
        if pulse_job is None:
            raise QiskitError('Invalid input object %s, must be either a QuantumCircuit, Schedule, or a list of either' % circuits)
        if pulse_job:
            raise QiskitError('Pulse simulation is currently not supported for fake backends.')
        if not _optionals.HAS_AER:
            warnings.warn('Aer not found using BasicAer and no noise', RuntimeWarning)
        if self.sim is None:
            self._setup_sim()
        self.sim._options = self._options
        job = self.sim.run(circuits, **kwargs)
        return job