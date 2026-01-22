import inspect
import warnings
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian, SProd, Prod, Sum
def _generator_backcompatibility(op):
    """Preserve backwards compatibility behaviour for PennyLane
    versions <=0.22, where generators returned List[type or ndarray, float].
    This function raises a deprecation warning, and converts to the new
    format where an instantiated Operator is returned."""
    warnings.warn('The Operator.generator property is deprecated. Please update the operator so that \n\t1. Operator.generator() is a method, and\n\t2. Operator.generator() returns an Operator instance representing the operator.', qml.PennyLaneDeprecationWarning)
    gen = op.generator
    if inspect.isclass(gen[0]):
        return gen[1] * gen[0](wires=op.wires)
    if isinstance(gen[0], np.ndarray) and len(gen[0].shape) == 2:
        return gen[1] * qml.Hermitian(gen[0], wires=op.wires)
    raise qml.operation.GeneratorUndefinedError