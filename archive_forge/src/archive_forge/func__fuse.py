from functools import partial
import numpy as np
from pennylane.math import abs as math_abs
from pennylane.math import sum as math_sum
from pennylane.math import allclose, arccos, arctan2, cos, get_interface, is_abstract, sin, stack
from pennylane.wires import Wires
from pennylane.ops.identity import GlobalPhase
def _fuse(angles_1, angles_2, abstract_jax=False):
    """Perform fusion of two angle sets. Separated out so we can do JIT with conditionals."""
    qw, qx, qy, qz = _quaternion_product(_zyz_to_quat(angles_1), _zyz_to_quat(angles_2))
    y_arg = 1 - 2 * (qx ** 2 + qy ** 2)
    if abstract_jax:
        from jax.lax import cond
        return cond(math_abs(y_arg) >= 1, partial(_singular_quat_to_zyz, abstract_jax=True), _regular_quat_to_zyz, qw, qx, qy, qz, y_arg)
    if abs(y_arg) >= 1:
        return _singular_quat_to_zyz(qw, qx, qy, qz, y_arg)
    return _regular_quat_to_zyz(qw, qx, qy, qz, y_arg)