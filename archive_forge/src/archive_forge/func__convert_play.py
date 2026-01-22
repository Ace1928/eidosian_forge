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
@_convert_instruction.register(instructions.Play)
def _convert_play(self, instruction, time_offset: int) -> PulseQobjInstruction:
    """Return converted `Play`.

        Args:
            instruction: Qiskit Pulse play instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
    if isinstance(instruction.pulse, library.SymbolicPulse):
        params = dict(instruction.pulse.parameters)
        if 'amp' in params and 'angle' in params:
            params['amp'] = complex(params['amp'] * np.exp(1j * params['angle']))
            del params['angle']
        command_dict = {'name': 'parametric_pulse', 'pulse_shape': ParametricPulseShapes.from_instance(instruction.pulse).name, 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'parameters': params}
    else:
        command_dict = {'name': instruction.name, 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name}
    return self._qobj_model(**command_dict)