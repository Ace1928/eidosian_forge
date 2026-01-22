import hashlib
import re
import warnings
from enum import Enum
from functools import singledispatchmethod
from typing import Union, List, Iterator, Optional
import numpy as np
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption, PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel
class QobjToInstructionConverter:
    """Converts Qobj data into Qiskit Pulse in-memory representation.

    This converter converts data from transfer layer into the in-memory representation of
    the front-end of Qiskit Pulse.

    The transfer layer format must be the text representation that coforms to
    the `OpenPulse specification<https://arxiv.org/abs/1809.03452>`__.
    Extention to the OpenPulse can be achieved by subclassing this this with
    extra methods corresponding to each augumented instruction. For example,

    .. code-block:: python

        class MyConverter(QobjToInstructionConverter):

            def get_supported_instructions(self):
                instructions = super().get_supported_instructions()
                instructions += ["new_inst"]

                return instructions

            def _convert_new_inst(self, instruction):
                return NewInstruction(...)

    where ``NewInstruction`` must be a subclass of :class:`~qiskit.pulse.instructions.Instruction`.
    """
    __chan_regex__ = re.compile('([a-zA-Z]+)(\\d+)')

    def __init__(self, pulse_library: Optional[List[PulseLibraryItem]]=None, **run_config):
        """Create new converter.

        Args:
            pulse_library: Pulse library in Qobj format.
             run_config: Run configuration.
        """
        pulse_library_dict = {}
        for lib_item in pulse_library:
            pulse_library_dict[lib_item.name] = lib_item.samples
        self._pulse_library = pulse_library_dict
        self._run_config = run_config

    def __call__(self, instruction: PulseQobjInstruction) -> Schedule:
        """Convert Qobj instruction to Qiskit in-memory representation.

        Args:
            instruction: Instruction data in Qobj format.

        Returns:
            Scheduled Qiskit Pulse instruction in Schedule format.
        """
        schedule = Schedule()
        for inst in self._get_sequences(instruction):
            schedule.insert(instruction.t0, inst, inplace=True)
        return schedule

    def _get_sequences(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """A method to iterate over pulse instructions without creating Schedule.

        .. note::

            This is internal fast-path function, and callers other than this converter class
            might directly use this method to generate schedule from multiple
            Qobj instructions. Because __call__ always returns a schedule with the time offset
            parsed instruction, composing multiple Qobj instructions to create
            a gate schedule is somewhat inefficient due to composing overhead of schedules.
            Directly combining instructions with this method is much performant.

        Args:
            instruction: Instruction data in Qobj format.

        Yields:
            Qiskit Pulse instructions.

        :meta public:
        """
        try:
            method = getattr(self, f'_convert_{instruction.name}')
        except AttributeError:
            method = self._convert_generic
        yield from method(instruction)

    def get_supported_instructions(self) -> List[str]:
        """Retrun a list of supported instructions."""
        return ['acquire', 'setp', 'fc', 'setf', 'shiftf', 'delay', 'parametric_pulse', 'snapshot']

    def get_channel(self, channel: str) -> channels.PulseChannel:
        """Parse and retrieve channel from ch string.

        Args:
            channel: String identifier of pulse instruction channel.

        Returns:
            Matched channel object.

        Raises:
            QiskitError: Is raised if valid channel is not matched
        """
        match = self.__chan_regex__.match(channel)
        if match:
            prefix, index = (match.group(1), int(match.group(2)))
            if prefix == channels.DriveChannel.prefix:
                return channels.DriveChannel(index)
            elif prefix == channels.MeasureChannel.prefix:
                return channels.MeasureChannel(index)
            elif prefix == channels.ControlChannel.prefix:
                return channels.ControlChannel(index)
        raise QiskitError('Channel %s is not valid' % channel)

    @staticmethod
    def disassemble_value(value_expr: Union[float, str]) -> Union[float, ParameterExpression]:
        """A helper function to format instruction operand.

        If parameter in string representation is specified, this method parses the
        input string and generates Qiskit ParameterExpression object.

        Args:
            value_expr: Operand value in Qobj.

        Returns:
            Parsed operand value. ParameterExpression object is returned if value is not number.
        """
        if isinstance(value_expr, str):
            str_expr = parse_string_expr(value_expr, partial_binding=False)
            value_expr = str_expr(**{pname: Parameter(pname) for pname in str_expr.params})
        return value_expr

    def _convert_acquire(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Return converted `Acquire` instruction.

        Args:
            instruction: Acquire qobj

        Yields:
            Qiskit Pulse acquire instructions
        """
        duration = instruction.duration
        qubits = instruction.qubits
        acquire_channels = [channels.AcquireChannel(qubit) for qubit in qubits]
        mem_slots = [channels.MemorySlot(instruction.memory_slot[i]) for i in range(len(qubits))]
        if hasattr(instruction, 'register_slot'):
            register_slots = [channels.RegisterSlot(instruction.register_slot[i]) for i in range(len(qubits))]
        else:
            register_slots = [None] * len(qubits)
        discriminators = instruction.discriminators if hasattr(instruction, 'discriminators') else None
        if not isinstance(discriminators, list):
            discriminators = [discriminators]
        if any((discriminators[i] != discriminators[0] for i in range(len(discriminators)))):
            warnings.warn('Can currently only support one discriminator per acquire. Defaulting to first discriminator entry.')
        discriminator = discriminators[0]
        if discriminator:
            discriminator = Discriminator(name=discriminators[0].name, **discriminators[0].params)
        kernels = instruction.kernels if hasattr(instruction, 'kernels') else None
        if not isinstance(kernels, list):
            kernels = [kernels]
        if any((kernels[0] != kernels[i] for i in range(len(kernels)))):
            warnings.warn('Can currently only support one kernel per acquire. Defaulting to first kernel entry.')
        kernel = kernels[0]
        if kernel:
            kernel = Kernel(name=kernels[0].name, **kernels[0].params)
        for acquire_channel, mem_slot, reg_slot in zip(acquire_channels, mem_slots, register_slots):
            yield instructions.Acquire(duration, acquire_channel, mem_slot=mem_slot, reg_slot=reg_slot, kernel=kernel, discriminator=discriminator)

    def _convert_setp(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Return converted `SetPhase` instruction.

        Args:
            instruction: SetPhase qobj instruction

        Yields:
            Qiskit Pulse set phase instructions
        """
        channel = self.get_channel(instruction.ch)
        phase = self.disassemble_value(instruction.phase)
        yield instructions.SetPhase(phase, channel)

    def _convert_fc(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Return converted `ShiftPhase` instruction.

        Args:
            instruction: ShiftPhase qobj instruction

        Yields:
            Qiskit Pulse shift phase schedule instructions
        """
        channel = self.get_channel(instruction.ch)
        phase = self.disassemble_value(instruction.phase)
        yield instructions.ShiftPhase(phase, channel)

    def _convert_setf(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Return converted `SetFrequencyInstruction` instruction.

        .. note::

            We assume frequency value is expressed in string with "GHz".
            Operand value is thus scaled by a factor of 1e9.

        Args:
            instruction: SetFrequency qobj instruction

        Yields:
            Qiskit Pulse set frequency instructions
        """
        channel = self.get_channel(instruction.ch)
        frequency = self.disassemble_value(instruction.frequency) * 1000000000.0
        yield instructions.SetFrequency(frequency, channel)

    def _convert_shiftf(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Return converted `ShiftFrequency` instruction.

        .. note::

            We assume frequency value is expressed in string with "GHz".
            Operand value is thus scaled by a factor of 1e9.

        Args:
            instruction: ShiftFrequency qobj instruction

        Yields:
            Qiskit Pulse shift frequency schedule instructions
        """
        channel = self.get_channel(instruction.ch)
        frequency = self.disassemble_value(instruction.frequency) * 1000000000.0
        yield instructions.ShiftFrequency(frequency, channel)

    def _convert_delay(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Return converted `Delay` instruction.

        Args:
            instruction: Delay qobj instruction

        Yields:
            Qiskit Pulse delay instructions
        """
        channel = self.get_channel(instruction.ch)
        duration = instruction.duration
        yield instructions.Delay(duration, channel)

    def _convert_parametric_pulse(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Return converted `Play` instruction with parametric pulse operand.

        .. note::

            If parametric pulse label is not provided by the backend, this method naively generates
            a pulse name based on the pulse shape and bound parameters. This pulse name is formatted
            to, for example, `gaussian_a4e3`, here the last four digits are a part of
            the hash string generated based on the pulse shape and the parameters.
            Because we are using a truncated hash for readability,
            there may be a small risk of pulse name collision with other pulses.
            Basically the parametric pulse name is used just for visualization purpose and
            the pulse module should not have dependency on the parametric pulse names.

        Args:
            instruction: Play qobj instruction with parametric pulse

        Yields:
            Qiskit Pulse play schedule instructions
        """
        channel = self.get_channel(instruction.ch)
        try:
            pulse_name = instruction.label
        except AttributeError:
            sorted_params = sorted(instruction.parameters.items(), key=lambda x: x[0])
            base_str = '{pulse}_{params}'.format(pulse=instruction.pulse_shape, params=str(sorted_params))
            short_pulse_id = hashlib.md5(base_str.encode('utf-8')).hexdigest()[:4]
            pulse_name = f'{instruction.pulse_shape}_{short_pulse_id}'
        params = dict(instruction.parameters)
        if 'amp' in params and isinstance(params['amp'], complex):
            params['angle'] = np.angle(params['amp'])
            params['amp'] = np.abs(params['amp'])
        pulse = ParametricPulseShapes.to_type(instruction.pulse_shape)(**params, name=pulse_name)
        yield instructions.Play(pulse, channel)

    def _convert_snapshot(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Return converted `Snapshot` instruction.

        Args:
            instruction: Snapshot qobj instruction

        Yields:
            Qiskit Pulse snapshot instructions
        """
        yield instructions.Snapshot(instruction.label, instruction.type)

    def _convert_generic(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        """Convert generic pulse instruction.

        Args:
            instruction: Generic qobj instruction

        Yields:
            Qiskit Pulse generic instructions

        Raises:
            QiskitError: When instruction name not found.
        """
        if instruction.name in self._pulse_library:
            waveform = library.Waveform(samples=self._pulse_library[instruction.name], name=instruction.name)
            channel = self.get_channel(instruction.ch)
            yield instructions.Play(waveform, channel)
        else:
            if (qubits := getattr(instruction, 'qubits', None)):
                msg = f'qubits {qubits}'
            else:
                msg = f'channel {instruction.ch}'
            raise QiskitError(f'Instruction {instruction.name} on {msg} is not found in Qiskit namespace. This instruction cannot be deserialized.')