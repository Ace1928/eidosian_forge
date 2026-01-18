from typing import Sequence, Callable
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.ops.qubit import Rot
from pennylane.math import allclose, stack, is_abstract
from pennylane.queuing import QueuingManager
from .optimization_utils import find_next_gate, fuse_rot_angles
A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        