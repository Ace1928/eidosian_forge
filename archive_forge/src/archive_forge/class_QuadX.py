import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class QuadX(CVObservable):
    """
    The position quadrature observable :math:`\\hat{x}`.

    When used with the :func:`~pennylane.expval` function, the position expectation
    value :math:`\\braket{\\hat{x}}` is returned. This corresponds to
    the mean displacement in the phase space along the :math:`x` axis.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Observable order: 1st order in the quadrature operators
    * Heisenberg representation:

      .. math:: d = [0, 1, 0]

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on
    """
    num_params = 0
    num_wires = 1
    ev_order = 1

    def __init__(self, wires):
        super().__init__(wires=wires)

    @staticmethod
    def _heisenberg_rep(p):
        return np.array([0, 1, 0])