import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
class QasmQobjConfig(SimpleNamespace):
    """A configuration for an OpenQASM 2 Qobj."""

    def __init__(self, shots=None, seed_simulator=None, memory=None, parameter_binds=None, meas_level=None, meas_return=None, memory_slots=None, n_qubits=None, pulse_library=None, calibrations=None, rep_delay=None, qubit_lo_freq=None, meas_lo_freq=None, **kwargs):
        """Model for RunConfig.

        Args:
            shots (int): the number of shots.
            seed_simulator (int): the seed to use in the simulator
            memory (bool): whether to request memory from backend (per-shot readouts)
            parameter_binds (list[dict]): List of parameter bindings
            meas_level (int): Measurement level 0, 1, or 2
            meas_return (str): For measurement level < 2, whether single or avg shots are returned
            memory_slots (int): The number of memory slots on the device
            n_qubits (int): The number of qubits on the device
            pulse_library (list): List of :class:`PulseLibraryItem`.
            calibrations (QasmExperimentCalibrations): Information required for Pulse gates.
            rep_delay (float): Delay between programs in sec. Only supported on certain
                backends (``backend.configuration().dynamic_reprate_enabled`` ). Must be from the
                range supplied by the backend (``backend.configuration().rep_delay_range``). Default
                is ``backend.configuration().default_rep_delay``.
            qubit_lo_freq (list): List of frequencies (as floats) for the qubit driver LO's in GHz.
            meas_lo_freq (list): List of frequencies (as floats) for the measurement driver LO's in
                GHz.
            kwargs: Additional free form key value fields to add to the
                configuration.
        """
        if shots is not None:
            self.shots = int(shots)
        if seed_simulator is not None:
            self.seed_simulator = int(seed_simulator)
        if memory is not None:
            self.memory = bool(memory)
        if parameter_binds is not None:
            self.parameter_binds = parameter_binds
        if meas_level is not None:
            self.meas_level = meas_level
        if meas_return is not None:
            self.meas_return = meas_return
        if memory_slots is not None:
            self.memory_slots = memory_slots
        if n_qubits is not None:
            self.n_qubits = n_qubits
        if pulse_library is not None:
            self.pulse_library = pulse_library
        if calibrations is not None:
            self.calibrations = calibrations
        if rep_delay is not None:
            self.rep_delay = rep_delay
        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq
        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq
        if kwargs:
            self.__dict__.update(kwargs)

    def to_dict(self):
        """Return a dictionary format representation of the OpenQASM 2 Qobj config.

        Returns:
            dict: The dictionary form of the QasmQobjConfig.
        """
        out_dict = copy.copy(self.__dict__)
        if hasattr(self, 'pulse_library'):
            out_dict['pulse_library'] = [x.to_dict() for x in self.pulse_library]
        if hasattr(self, 'calibrations'):
            out_dict['calibrations'] = self.calibrations.to_dict()
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjConfig object from a dictionary.

        Args:
            data (dict): A dictionary for the config

        Returns:
            QasmQobjConfig: The object from the input dictionary.
        """
        if 'pulse_library' in data:
            pulse_lib = data.pop('pulse_library')
            pulse_lib_obj = [PulseLibraryItem.from_dict(x) for x in pulse_lib]
            data['pulse_library'] = pulse_lib_obj
        if 'calibrations' in data:
            calibrations = data.pop('calibrations')
            data['calibrations'] = QasmExperimentCalibrations.from_dict(calibrations)
        return cls(**data)

    def __eq__(self, other):
        if isinstance(other, QasmQobjConfig):
            if self.to_dict() == other.to_dict():
                return True
        return False