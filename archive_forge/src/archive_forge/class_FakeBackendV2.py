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
class FakeBackendV2(BackendV2):
    """A fake backend class for testing and noisy simulation using real backend
    snapshots.

    The class inherits :class:`~qiskit.providers.BackendV2` class. This version
    differs from earlier :class:`~qiskit.providers.fake_provider.FakeBackend` (V1) class in a
    few aspects. Firstly, configuration attribute no longer exsists. Instead,
    attributes exposing equivalent required immutable properties of the backend
    device are added. For example ``fake_backend.configuration().n_qubits`` is
    accessible from ``fake_backend.num_qubits`` now. Secondly, this version
    removes extra abstractions :class:`~qiskit.providers.fake_provider.FakeQasmBackend` and
    :class:`~qiskit.providers.fake_provider.FakePulseBackend` that were present in V1.
    """
    dirname = None
    conf_filename = None
    props_filename = None
    defs_filename = None
    backend_name = None

    def __init__(self):
        """FakeBackendV2 initializer."""
        self._conf_dict = self._get_conf_dict_from_json()
        self._props_dict = None
        self._defs_dict = None
        super().__init__(provider=None, name=self._conf_dict.get('backend_name'), description=self._conf_dict.get('description'), online_date=self._conf_dict.get('online_date'), backend_version=self._conf_dict.get('backend_version'))
        self._target = None
        self.sim = None
        if 'channels' in self._conf_dict:
            self._parse_channels(self._conf_dict['channels'])

    def _parse_channels(self, channels):
        type_map = {'acquire': pulse.AcquireChannel, 'drive': pulse.DriveChannel, 'measure': pulse.MeasureChannel, 'control': pulse.ControlChannel}
        identifier_pattern = re.compile('\\D+(?P<index>\\d+)')
        channels_map = {'acquire': collections.defaultdict(list), 'drive': collections.defaultdict(list), 'measure': collections.defaultdict(list), 'control': collections.defaultdict(list)}
        for identifier, spec in channels.items():
            channel_type = spec['type']
            out = re.match(identifier_pattern, identifier)
            if out is None:
                continue
            channel_index = int(out.groupdict()['index'])
            qubit_index = tuple(spec['operates']['qubits'])
            chan_obj = type_map[channel_type](channel_index)
            channels_map[channel_type][qubit_index].append(chan_obj)
        setattr(self, 'channels_map', channels_map)

    def _setup_sim(self):
        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator
            self.sim = AerSimulator()
            if self.target and self._props_dict:
                noise_model = self._get_noise_model_from_backend_v2()
                self.sim.set_options(noise_model=noise_model)
                self.set_options(noise_model=noise_model)
        else:
            self.sim = basic_provider.BasicSimulator()

    def _get_conf_dict_from_json(self):
        if not self.conf_filename:
            return None
        conf_dict = self._load_json(self.conf_filename)
        decode_backend_configuration(conf_dict)
        conf_dict['backend_name'] = self.backend_name
        return conf_dict

    def _set_props_dict_from_json(self):
        if self.props_filename:
            props_dict = self._load_json(self.props_filename)
            decode_backend_properties(props_dict)
            self._props_dict = props_dict

    def _set_defs_dict_from_json(self):
        if self.defs_filename:
            defs_dict = self._load_json(self.defs_filename)
            decode_pulse_defaults(defs_dict)
            self._defs_dict = defs_dict

    def _load_json(self, filename: str) -> dict:
        with open(os.path.join(self.dirname, filename)) as f_json:
            the_json = json.load(f_json)
        return the_json

    @property
    def target(self) -> Target:
        """A :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: Target
        """
        if self._target is None:
            self._get_conf_dict_from_json()
            if self._props_dict is None:
                self._set_props_dict_from_json()
            if self._defs_dict is None:
                self._set_defs_dict_from_json()
            conf = BackendConfiguration.from_dict(self._conf_dict)
            props = None
            if self._props_dict is not None:
                props = BackendProperties.from_dict(self._props_dict)
            defaults = None
            if self._defs_dict is not None:
                defaults = PulseDefaults.from_dict(self._defs_dict)
            self._target = convert_to_target(configuration=conf, properties=props, defaults=defaults)
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
                default values set
        """
        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator
            return AerSimulator._default_options()
        else:
            return basic_provider.BasicSimulator._default_options()

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals

        Returns:
            The output signal timestep in seconds.
        """
        dtm = self._conf_dict.get('dtm')
        if dtm is not None:
            return dtm * 1e-09
        else:
            return None

    @property
    def meas_map(self) -> List[List[int]]:
        """Return the grouping of measurements which are multiplexed
        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            The grouping of measurements which are multiplexed
        """
        return self._conf_dict.get('meas_map')

    def drive_channel(self, qubit: int):
        """Return the drive channel for the given qubit.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            DriveChannel: The Qubit drive channel
        """
        drive_channels_map = getattr(self, 'channels_map', {}).get('drive', {})
        qubits = (qubit,)
        if qubits in drive_channels_map:
            return drive_channels_map[qubits][0]
        return None

    def measure_channel(self, qubit: int):
        """Return the measure stimulus channel for the given qubit.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            MeasureChannel: The Qubit measurement stimulus line
        """
        measure_channels_map = getattr(self, 'channels_map', {}).get('measure', {})
        qubits = (qubit,)
        if qubits in measure_channels_map:
            return measure_channels_map[qubits][0]
        return None

    def acquire_channel(self, qubit: int):
        """Return the acquisition channel for the given qubit.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            AcquireChannel: The Qubit measurement acquisition line.
        """
        acquire_channels_map = getattr(self, 'channels_map', {}).get('acquire', {})
        qubits = (qubit,)
        if qubits in acquire_channels_map:
            return acquire_channels_map[qubits][0]
        return None

    def control_channel(self, qubits: Iterable[int]):
        """Return the secondary drive channel for the given qubit

        This is typically utilized for controlling multiqubit interactions.
        This channel is derived from other channels.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Args:
            qubits: Tuple or list of qubits of the form
                ``(control_qubit, target_qubit)``.

        Returns:
            List[ControlChannel]: The multi qubit control line.
        """
        control_channels_map = getattr(self, 'channels_map', {}).get('control', {})
        qubits = tuple(qubits)
        if qubits in control_channels_map:
            return control_channels_map[qubits]
        return []

    def run(self, run_input, **options):
        """Run on the fake backend using a simulator.

        This method runs circuit jobs (an individual or a list of QuantumCircuit
        ) and pulse jobs (an individual or a list of Schedule or ScheduleBlock)
        using a :class:`.BasicSimulator` or Aer simulator and returns a
        :class:`~qiskit.providers.Job` object.

        If qiskit-aer is installed, jobs will be run using AerSimulator with
        noise model of the fake backend. Otherwise, jobs will be run using
        :class:`.BasicSimulator` without noise.

        Currently noisy simulation of a pulse job is not supported yet in
        FakeBackendV2.

        Args:
            run_input (QuantumCircuit or Schedule or ScheduleBlock or list): An
                individual or a list of
                :class:`~qiskit.circuit.QuantumCircuit`,
                :class:`~qiskit.pulse.ScheduleBlock`, or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.

        Returns:
            Job: The job object for the run

        Raises:
            QiskitError: If a pulse job is supplied and qiskit-aer is not installed.
        """
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
            warnings.warn('Aer not found using BasicProvider and no noise', RuntimeWarning)
        if self.sim is None:
            self._setup_sim()
        self.sim._options = self._options
        job = self.sim.run(circuits, **options)
        return job

    def _get_noise_model_from_backend_v2(self, gate_error=True, readout_error=True, thermal_relaxation=True, temperature=0, gate_lengths=None, gate_length_units='ns'):
        """Build noise model from BackendV2.

        This is a temporary fix until qiskit-aer supports building noise model
        from a BackendV2 object.
        """
        from qiskit.circuit import Delay
        from qiskit.providers.exceptions import BackendPropertyError
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.noise.device.models import _excited_population, basic_device_gate_errors, basic_device_readout_errors
        from qiskit_aer.noise.passes import RelaxationNoisePass
        if self._props_dict is None:
            self._set_props_dict_from_json()
        properties = BackendProperties.from_dict(self._props_dict)
        basis_gates = self.operation_names
        num_qubits = self.num_qubits
        dt = self.dt
        noise_model = NoiseModel(basis_gates=basis_gates)
        if readout_error:
            for qubits, error in basic_device_readout_errors(properties):
                noise_model.add_readout_error(error, qubits)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='qiskit_aer.noise.device.models')
            gate_errors = basic_device_gate_errors(properties, gate_error=gate_error, thermal_relaxation=thermal_relaxation, gate_lengths=gate_lengths, gate_length_units=gate_length_units, temperature=temperature)
        for name, qubits, error in gate_errors:
            noise_model.add_quantum_error(error, name, qubits)
        if thermal_relaxation:
            try:
                excited_state_populations = [_excited_population(freq=properties.frequency(q), temperature=temperature) for q in range(num_qubits)]
            except BackendPropertyError:
                excited_state_populations = None
            try:
                delay_pass = RelaxationNoisePass(t1s=[properties.t1(q) for q in range(num_qubits)], t2s=[properties.t2(q) for q in range(num_qubits)], dt=dt, op_types=Delay, excited_state_populations=excited_state_populations)
                noise_model._custom_noise_passes.append(delay_pass)
            except BackendPropertyError:
                pass
        return noise_model