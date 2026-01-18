from types import FunctionType
from typing import Type, Union, Callable, Sequence
import pennylane as qml
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.ops.op_math import Adjoint
A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        