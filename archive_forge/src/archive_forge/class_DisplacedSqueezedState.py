import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class DisplacedSqueezedState(CVOperation):
    """
    Prepares a displaced squeezed vacuum state.

    A displaced squeezed state is prepared by squeezing a vacuum state, and
    then applying a displacement operator,

    .. math::
       \\ket{\\alpha,z} = D(\\alpha)\\ket{0,z} = D(\\alpha)S(z)\\ket{0},

    with the displacement parameter :math:`\\alpha=ae^{i\\phi_a}` and the squeezing parameter :math:`z=re^{i\\phi_r}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 4
    * Gradient recipe: None (uses finite difference)

    Args:
        a (float): displacement magnitude :math:`a=|\\alpha|`
        phi_a (float): displacement angle :math:`\\phi_a`
        r (float): squeezing magnitude :math:`r=|z|`
        phi_r (float): squeezing angle :math:`\\phi_r`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 4
    num_wires = 1
    grad_method = 'F'

    def __init__(self, a, phi_a, r, phi_r, wires, id=None):
        super().__init__(a, phi_a, r, phi_r, wires=wires, id=id)