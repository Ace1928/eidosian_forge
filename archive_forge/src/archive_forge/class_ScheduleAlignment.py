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
class ScheduleAlignment(TypeKeyBase):
    """Type key enum for schedule block alignment context object."""
    LEFT = b'l'
    RIGHT = b'r'
    SEQUENTIAL = b's'
    EQUISPACED = b'e'

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, AlignLeft):
            return cls.LEFT
        if isinstance(obj, AlignRight):
            return cls.RIGHT
        if isinstance(obj, AlignSequential):
            return cls.SEQUENTIAL
        if isinstance(obj, AlignEquispaced):
            return cls.EQUISPACED
        raise exceptions.QpyError(f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        if type_key == cls.LEFT:
            return AlignLeft
        if type_key == cls.RIGHT:
            return AlignRight
        if type_key == cls.SEQUENTIAL:
            return AlignSequential
        if type_key == cls.EQUISPACED:
            return AlignEquispaced
        raise exceptions.QpyError(f"A class corresponding to type key '{type_key}' is not found in {cls.__name__} namespace.")