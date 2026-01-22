from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.gradients.metric_tensor import _get_aux_wire
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable_hadamard_gradient
from .gradient_transform import (
def _get_generators(trainable_op):
    """From a trainable operation, extract the unitary generators and their coefficients. If an operation is added here
    one needs to also update the list of supported operation in the expand function given to the gradient transform.
    """
    if isinstance(trainable_op, (qml.PhaseShift, qml.U1)):
        generators = [qml.Z(trainable_op.wires)]
        coeffs = [-0.5]
    elif isinstance(trainable_op, qml.CRX):
        generators = [qml.X(trainable_op.wires[1]), qml.prod(qml.Z(trainable_op.wires[0]), qml.X(trainable_op.wires[1]))]
        coeffs = [-0.25, 0.25]
    elif isinstance(trainable_op, qml.CRY):
        generators = [qml.Y(trainable_op.wires[1]), qml.prod(qml.Z(trainable_op.wires[0]), qml.Y(trainable_op.wires[1]))]
        coeffs = [-0.25, 0.25]
    elif isinstance(trainable_op, qml.CRZ):
        generators = [qml.Z(trainable_op.wires[1]), qml.prod(qml.Z(trainable_op.wires[0]), qml.Z(trainable_op.wires[1]))]
        coeffs = [-0.25, 0.25]
    elif isinstance(trainable_op, qml.IsingXX):
        generators = [qml.prod(qml.X(trainable_op.wires[0]), qml.X(trainable_op.wires[1]))]
        coeffs = [-0.5]
    elif isinstance(trainable_op, qml.IsingYY):
        generators = [qml.prod(qml.Y(trainable_op.wires[0]), qml.Y(trainable_op.wires[1]))]
        coeffs = [-0.5]
    elif isinstance(trainable_op, qml.IsingZZ):
        generators = [qml.prod(qml.Z(trainable_op.wires[0]), qml.Z(trainable_op.wires[1]))]
        coeffs = [-0.5]
    elif isinstance(trainable_op, qml.Rot):
        generators = [qml.Z(trainable_op.wires)]
        coeffs = [-0.5]
    else:
        generators = trainable_op.generator().ops
        coeffs = trainable_op.generator().coeffs
    return (coeffs, generators)