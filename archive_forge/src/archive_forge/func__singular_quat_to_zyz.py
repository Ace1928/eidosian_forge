from functools import partial
import numpy as np
from pennylane.math import abs as math_abs
from pennylane.math import sum as math_sum
from pennylane.math import allclose, arccos, arctan2, cos, get_interface, is_abstract, sin, stack
from pennylane.wires import Wires
from pennylane.ops.identity import GlobalPhase
def _singular_quat_to_zyz(qw, qx, qy, qz, y_arg, abstract_jax=False):
    """Compute the ZYZ angles for the singular case of qx = qy = 0"""
    z1_arg1 = 2 * (qx * qy + qz * qw)
    z1_arg2 = 1 - 2 * (qx ** 2 + qz ** 2)
    if abstract_jax:
        from jax.lax import cond
        return cond(y_arg > 0, lambda z1_arg1, z1_arg2: stack([arctan2(z1_arg1, z1_arg2), 0.0, 0.0]), lambda z1_arg1, z1_arg2: stack([-arctan2(z1_arg1, z1_arg2), np.pi, 0.0]), z1_arg1, z1_arg2)
    if y_arg > 0:
        z1 = arctan2(z1_arg1, z1_arg2)
        y = z2 = 0.0
    else:
        z1 = -arctan2(z1_arg1, z1_arg2)
        y = np.pi
        z2 = 0.0
    return stack([z1, y, z2])