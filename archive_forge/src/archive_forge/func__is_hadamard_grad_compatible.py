import contextlib
import pennylane as qml
from pennylane.operation import (
@qml.BooleanFn
def _is_hadamard_grad_compatible(obj):
    """Check if the operation is compatible with Hadamard gradient transform."""
    return obj.name in hadamard_comp_list