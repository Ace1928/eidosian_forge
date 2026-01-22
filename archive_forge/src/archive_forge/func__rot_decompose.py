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
def _rot_decompose(op):
    """Decomposes a rotation operation: :class:`~.Rot`, :class:`~.RX`, :class:`~.RY`, :class:`~.RZ`,
    :class:`~.PhaseShift` into a basis composed of :class:`~.RZ`, :class:`~.S`, and :class:`~.Hadamard`.
    """
    d_ops = []
    if isinstance(op, qml.Rot):
        (phi, theta, omega), wires = (op.parameters, op.wires)
        for dec in [qml.RZ(phi, wires), qml.RY(theta, wires), qml.RZ(omega, wires)]:
            d_ops.extend(_rot_decompose(dec))
        return d_ops
    (theta,), wires = (op.data, op.wires)
    if isinstance(op, qml.ops.Adjoint):
        ops_ = _rot_decompose(op.base.adjoint())
    elif isinstance(op, qml.RX):
        ops_ = _simplify_param(theta, qml.X(wires))
        if ops_ is None:
            ops_ = [qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires)]
    elif isinstance(op, qml.RY):
        ops_ = _simplify_param(theta, qml.Y(wires))
        if ops_ is None:
            ops_ = [qml.S(wires), qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires), qml.adjoint(qml.S(wires))][::-1]
    elif isinstance(op, qml.RZ):
        ops_ = _simplify_param(theta, qml.Z(wires))
        if ops_ is None:
            ops_ = [qml.RZ(theta, wires)]
    elif isinstance(op, qml.PhaseShift):
        ops_ = _simplify_param(theta, qml.Z(wires))
        if ops_ is None:
            ops_ = [qml.RZ(theta, wires=wires), qml.GlobalPhase(-theta / 2)]
        else:
            ops_.append(qml.GlobalPhase(-theta / 2))
    else:
        raise ValueError(f'Operation {op} is not a supported Pauli rotation: qml.RX, qml.RY, qml.RZ, qml.Rot and qml.PhaseShift.')
    d_ops.extend(ops_)
    return d_ops