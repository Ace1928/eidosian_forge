import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class PolyXP(CVObservable):
    """
    An arbitrary second-order polynomial observable.

    Represents an arbitrary observable :math:`P(\\x,\\p)` that is a second order
    polynomial in the basis :math:`\\mathbf{r} = (\\I, \\x_0, \\p_0, \\x_1, \\p_1, \\ldots)`.

    For first-order observables the representation is a real vector
    :math:`\\mathbf{d}` such that :math:`P(\\x,\\p) = \\mathbf{d}^T \\mathbf{r}`.

    For second-order observables the representation is a real symmetric
    matrix :math:`A` such that :math:`P(\\x,\\p) = \\mathbf{r}^T A \\mathbf{r}`.

    Used for evaluating arbitrary order-2 CV expectation values of
    :class:`~.pennylane.tape.CVParamShiftTape`.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Observable order: 2nd order in the quadrature operators
    * Heisenberg representation: :math:`A`

    Args:
        q (array[float]): expansion coefficients
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    """
    num_params = 1
    num_wires = AnyWires
    grad_method = 'F'
    ev_order = 2

    def __init__(self, q, wires, id=None):
        super().__init__(q, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        return p[0]