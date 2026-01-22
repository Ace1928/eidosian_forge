from typing import Sequence, Callable
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.ops.qubit.attributes import (
from .optimization_utils import find_next_gate
A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        