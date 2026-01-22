import math
import warnings
from itertools import product
from typing import Sequence, Callable
import pennylane as qml
from pennylane.ops import Adjoint
from pennylane.queuing import QueuingManager
from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
from pennylane.transforms.optimization.optimization_utils import find_next_gate, _fuse_global_phases
from pennylane.ops.op_math.decompositions.solovay_kitaev import sk_decomposition
def _one_qubit_decompose(op):
    """Decomposition for single qubit operations using combination of :class:`~.RZ`, :class:`~.S`, and
    :class:`~.Hadamard`."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sd_ops = qml.ops.one_qubit_decomposition(qml.matrix(op), op.wires, 'ZXZ', return_global_phase=True)
    gphase_op = sd_ops.pop()
    d_ops = []
    for sd_op in sd_ops:
        d_ops.extend(_rot_decompose(sd_op) if sd_op.num_params else [sd_op])
    return (d_ops, gphase_op)