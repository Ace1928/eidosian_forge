from typing import Sequence, Callable
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
import pennylane as qml
from pennylane.ops.op_math.decompositions import one_qubit_decomposition, two_qubit_decomposition
A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        