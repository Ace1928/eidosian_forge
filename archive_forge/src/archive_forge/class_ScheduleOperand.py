from abc import abstractmethod
from enum import Enum, IntEnum
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.annotated_operation import AnnotatedOperation, Modifier
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.pulse.channels import (
from qiskit.pulse.configuration import Discriminator, Kernel
from qiskit.pulse.instructions import (
from qiskit.pulse.library import Waveform, SymbolicPulse
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.pulse.transforms.alignments import (
from qiskit.qpy import exceptions
class ScheduleOperand(TypeKeyBase):
    """Type key enum for schedule instruction operand object."""
    WAVEFORM = b'w'
    SYMBOLIC_PULSE = b's'
    CHANNEL = b'c'
    KERNEL = b'k'
    DISCRIMINATOR = b'd'
    OPERAND_STR = b'o'

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, Waveform):
            return cls.WAVEFORM
        if isinstance(obj, SymbolicPulse):
            return cls.SYMBOLIC_PULSE
        if isinstance(obj, Channel):
            return cls.CHANNEL
        if isinstance(obj, str):
            return cls.OPERAND_STR
        if isinstance(obj, Kernel):
            return cls.KERNEL
        if isinstance(obj, Discriminator):
            return cls.DISCRIMINATOR
        raise exceptions.QpyError(f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError