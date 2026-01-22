import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class TensorN(CVObservable):
    """
    The tensor product of the :class:`~.NumberOperator` acting on different wires.

    If a single wire is defined, returns a :class:`~.NumberOperator` instance for convenient gradient computations.

    When used with the :func:`~pennylane.expval` function, the expectation value
    :math:`\\langle \\hat{n}_{i_0} \\hat{n}_{i_1}\\dots \\hat{n}_{i_{N-1}}\\rangle`
    for a (sub)set of modes :math:`[i_0, i_1, \\dots, i_{N-1}]` of the system is
    returned.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 0

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on

    .. details::
        :title: Usage Details

        Example for multiple modes:

        >>> cv_obs = qml.TensorN(wires=[0, 1])
        >>> cv_obs
        TensorN(wires=[0, 1])
        >>> cv_obs.ev_order is None
        True

        Example for a single mode (yields a :class:`~.NumberOperator`):

        >>> cv_obs = qml.TensorN(wires=[1])
        >>> cv_obs
        NumberOperator(wires=[1])
        >>> cv_obs.ev_order
        2
    """
    num_params = 0
    num_wires = AnyWires
    ev_order = None

    def __init__(self, wires):
        super().__init__(wires=wires)

    def __new__(cls, wires=None):
        if wires is not None and (isinstance(wires, int) or len(wires) == 1):
            return NumberOperator(wires=wires)
        return super().__new__(cls)

    def label(self, decimals=None, base_label=None, cache=None):
        if base_label is not None:
            return base_label
        return '⊗'.join(('n' for _ in self.wires))