import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class CatState(CVOperation):
    """
    Prepares a cat state.

    A cat state is the coherent superposition of two coherent states,

    .. math::
       \\ket{\\text{cat}(\\alpha)} = \\frac{1}{N} (\\ket{\\alpha} +e^{ip\\pi} \\ket{-\\alpha}),

    where :math:`\\ket{\\pm\\alpha} = D(\\pm\\alpha)\\ket{0}` are coherent states with displacement
    parameters :math:`\\pm\\alpha=\\pm ae^{i\\phi}` and
    :math:`N = \\sqrt{2 (1+\\cos(p\\pi)e^{-2|\\alpha|^2})}` is the normalization factor.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Gradient recipe: None (uses finite difference)

    Args:
        a (float): displacement magnitude :math:`a=|\\alpha|`
        phi (float): displacement angle :math:`\\phi`
        p (float): parity, where :math:`p=0` corresponds to an even
            cat state, and :math:`p=1` an odd cat state.
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 3
    num_wires = 1
    grad_method = 'F'

    def __init__(self, a, phi, p, wires, id=None):
        super().__init__(a, phi, p, wires=wires, id=id)