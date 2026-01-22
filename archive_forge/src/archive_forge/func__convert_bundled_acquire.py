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
def _convert_bundled_acquire(self, instruction_bundle: List[instructions.Acquire], time_offset: int) -> PulseQobjInstruction:
    """Return converted list of parallel `Acquire` instructions.

        Args:
            instruction_bundle: List of Qiskit Pulse acquire instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.

        Raises:
            QiskitError: When instructions are not aligned.
            QiskitError: When instructions have different duration.
            QiskitError: When discriminator or kernel is missing in a part of instructions.
        """
    meas_level = self._run_config.get('meas_level', 2)
    t0 = instruction_bundle[0].start_time
    duration = instruction_bundle[0].duration
    memory_slots = []
    register_slots = []
    qubits = []
    discriminators = []
    kernels = []
    for instruction in instruction_bundle:
        qubits.append(instruction.channel.index)
        if instruction.start_time != t0:
            raise QiskitError('The supplied acquire instructions have different starting times. Something has gone wrong calling this code. Please report this issue.')
        if instruction.duration != duration:
            raise QiskitError('Acquire instructions beginning at the same time must have same duration.')
        if instruction.mem_slot:
            memory_slots.append(instruction.mem_slot.index)
        if meas_level == MeasLevel.CLASSIFIED:
            if instruction.discriminator:
                discriminators.append(QobjMeasurementOption(name=instruction.discriminator.name, params=instruction.discriminator.params))
            if instruction.reg_slot:
                register_slots.append(instruction.reg_slot.index)
        if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
            if instruction.kernel:
                kernels.append(QobjMeasurementOption(name=instruction.kernel.name, params=instruction.kernel.params))
    command_dict = {'name': 'acquire', 't0': time_offset + t0, 'duration': duration, 'qubits': qubits}
    if memory_slots:
        command_dict['memory_slot'] = memory_slots
    if register_slots:
        command_dict['register_slot'] = register_slots
    if discriminators:
        num_discriminators = len(discriminators)
        if num_discriminators == len(qubits) or num_discriminators == 1:
            command_dict['discriminators'] = discriminators
        else:
            raise QiskitError('A discriminator must be supplied for every acquisition or a single discriminator for all acquisitions.')
    if kernels:
        num_kernels = len(kernels)
        if num_kernels == len(qubits) or num_kernels == 1:
            command_dict['kernels'] = kernels
        else:
            raise QiskitError('A kernel must be supplied for every acquisition or a single kernel for all acquisitions.')
    return self._qobj_model(**command_dict)