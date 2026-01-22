from functools import partial
import numpy as np
from pennylane.math import abs as math_abs
from pennylane.math import sum as math_sum
from pennylane.math import allclose, arccos, arctan2, cos, get_interface, is_abstract, sin, stack
from pennylane.wires import Wires
from pennylane.ops.identity import GlobalPhase
def _fuse_global_phases(operations):
    """Fuse all the global phase operations into single one.

    Args:
        operations (list[Operation]): list of operations to be iterated over

    Returns:
        transformed list with a single :func:`~.pennylane.GlobalPhase` operation.
    """
    fused_ops, global_ops = ([], [])
    for op in operations:
        if isinstance(op, GlobalPhase):
            global_ops.append(op)
        else:
            fused_ops.append(op)
    fused_ops.append(GlobalPhase(math_sum((op.data[0] for op in global_ops))))
    return fused_ops