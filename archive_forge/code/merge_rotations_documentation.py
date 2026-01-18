from typing import Sequence, Callable
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.math import allclose, stack, cast_like, zeros, is_abstract, get_interface
from pennylane.queuing import QueuingManager
from pennylane.ops.qubit.attributes import composable_rotations
from pennylane.ops.op_math import Adjoint
from .optimization_utils import find_next_gate, fuse_rot_angles
A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        