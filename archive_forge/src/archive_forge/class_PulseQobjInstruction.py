import copy
import pprint
from typing import Union, List
import numpy
from qiskit.qobj.common import QobjDictField
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.common import QobjExperimentHeader
class PulseQobjInstruction:
    """A class representing a single instruction in an PulseQobj Experiment."""
    _COMMON_ATTRS = ['ch', 'conditional', 'val', 'phase', 'frequency', 'duration', 'qubits', 'memory_slot', 'register_slot', 'label', 'type', 'pulse_shape', 'parameters']

    def __init__(self, name, t0, ch=None, conditional=None, val=None, phase=None, duration=None, qubits=None, memory_slot=None, register_slot=None, kernels=None, discriminators=None, label=None, type=None, pulse_shape=None, parameters=None, frequency=None):
        """Instantiate a new PulseQobjInstruction object.

        Args:
            name (str): The name of the instruction
            t0 (int): Pulse start time in integer **dt** units.
            ch (str): The channel to apply the pulse instruction.
            conditional (int): The register to use for a conditional for this
                instruction
            val (complex): Complex value to apply, bounded by an absolute value
                of 1.
            phase (float): if a ``fc`` instruction, the frame change phase in
                radians.
            frequency (float): if a ``sf`` instruction, the frequency in Hz.
            duration (int): The duration of the pulse in **dt** units.
            qubits (list): A list of ``int`` representing the qubits the
                instruction operates on
            memory_slot (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of memory slots to store the
                measurement results in (must be the same length as qubits).
                If a ``bfunc`` instruction this is a single ``int`` of the
                memory slot to store the boolean function result in.
            register_slot (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of register slots in which to
                store the measurement results (must be the same length as
                qubits). If a ``bfunc`` instruction this is a single ``int``
                of the register slot in which to store the result.
            kernels (list): List of :class:`QobjMeasurementOption` objects
                defining the measurement kernels and set of parameters if the
                measurement level is 1 or 2. Only used for ``acquire``
                instructions.
            discriminators (list): A list of :class:`QobjMeasurementOption`
                used to set the discriminators to be used if the measurement
                level is 2. Only used for ``acquire`` instructions.
            label (str): Label of instruction
            type (str): Type of instruction
            pulse_shape (str): The shape of the parametric pulse
            parameters (dict): The parameters for a parametric pulse
        """
        self.name = name
        self.t0 = t0
        if ch is not None:
            self.ch = ch
        if conditional is not None:
            self.conditional = conditional
        if val is not None:
            self.val = val
        if phase is not None:
            self.phase = phase
        if frequency is not None:
            self.frequency = frequency
        if duration is not None:
            self.duration = duration
        if qubits is not None:
            self.qubits = qubits
        if memory_slot is not None:
            self.memory_slot = memory_slot
        if register_slot is not None:
            self.register_slot = register_slot
        if kernels is not None:
            self.kernels = kernels
        if discriminators is not None:
            self.discriminators = discriminators
        if label is not None:
            self.label = label
        if type is not None:
            self.type = type
        if pulse_shape is not None:
            self.pulse_shape = pulse_shape
        if parameters is not None:
            self.parameters = parameters

    def to_dict(self):
        """Return a dictionary format representation of the Instruction.

        Returns:
            dict: The dictionary form of the PulseQobjInstruction.
        """
        out_dict = {'name': self.name, 't0': self.t0}
        for attr in self._COMMON_ATTRS:
            if hasattr(self, attr):
                out_dict[attr] = getattr(self, attr)
        if hasattr(self, 'kernels'):
            out_dict['kernels'] = [x.to_dict() for x in self.kernels]
        if hasattr(self, 'discriminators'):
            out_dict['discriminators'] = [x.to_dict() for x in self.discriminators]
        return out_dict

    def __repr__(self):
        out = f'PulseQobjInstruction(name="{self.name}", t0={self.t0}'
        for attr in self._COMMON_ATTRS:
            attr_val = getattr(self, attr, None)
            if attr_val is not None:
                if isinstance(attr_val, str):
                    out += f', {attr}="{attr_val}"'
                else:
                    out += f', {attr}={attr_val}'
        out += ')'
        return out

    def __str__(self):
        out = 'Instruction: %s\n' % self.name
        out += '\t\tt0: %s\n' % self.t0
        for attr in self._COMMON_ATTRS:
            if hasattr(self, attr):
                out += f'\t\t{attr}: {getattr(self, attr)}\n'
        return out

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseQobjExperimentConfig object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            PulseQobjInstruction: The object from the input dictionary.
        """
        schema = {'discriminators': QobjMeasurementOption, 'kernels': QobjMeasurementOption}
        skip = ['t0', 'name']
        in_data = {}
        for key, value in data.items():
            if key in skip:
                continue
            if key == 'parameters':
                formatted_value = value.copy()
                if 'amp' in formatted_value:
                    formatted_value['amp'] = _to_complex(formatted_value['amp'])
                in_data[key] = formatted_value
                continue
            if key in schema:
                if isinstance(value, list):
                    in_data[key] = list(map(schema[key].from_dict, value))
                else:
                    in_data[key] = schema[key].from_dict(value)
            else:
                in_data[key] = value
        return cls(data['name'], data['t0'], **in_data)

    def __eq__(self, other):
        if isinstance(other, PulseQobjInstruction):
            if self.to_dict() == other.to_dict():
                return True
        return False