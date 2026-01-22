import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class FockStateProjector(CVObservable):
    """
    The number state observable :math:`\\ket{n}\\bra{n}`.

    Represents the non-Gaussian number state observable

    .. math:: \\ket{n}\\bra{n} = \\ket{n_0, n_1, \\dots, n_P}\\bra{n_0, n_1, \\dots, n_P}

    where :math:`n_i` is the occupation number of the :math:`i` th wire.

    The expectation of this observable is

    .. math::
        E[\\ket{n}\\bra{n}] = \\text{Tr}(\\ket{n}\\bra{n}\\rho)
        = \\text{Tr}(\\braketT{n}{\\rho}{n})
        = \\braketT{n}{\\rho}{n}

    corresponding to the probability of measuring the quantum state in the state
    :math:`\\ket{n}=\\ket{n_0, n_1, \\dots, n_P}`.

    .. note::

        If ``expval(FockStateProjector)`` is applied to a subset of wires,
        the unaffected wires are traced out prior to the expectation value
        calculation.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Observable order: None (non-Gaussian)

    Args:
        n (array): Array of non-negative integers representing the number state
            observable :math:`\\ket{n}\\bra{n}=\\ket{n_0, n_1, \\dots, n_P}\\bra{n_0, n_1, \\dots, n_P}`.

            For example, to return the observable :math:`\\ket{0,4,2}\\bra{0,4,2}` acting on
            wires 0, 1, and 3 of a QNode, you would call ``FockStateProjector(np.array([0, 4, 2], wires=[0, 1, 3]))``.

            Note that ``len(n)==len(wires)``, and that ``len(n)`` cannot exceed the
            total number of wires in the QNode.
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = AnyWires
    grad_method = None
    ev_order = None

    def __init__(self, n, wires, id=None):
        super().__init__(n, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        """A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> qml.FockStateProjector([1,2,3], wires=(0,1,2)).label()
        '|123⟩⟨123|'

        """
        if base_label is not None:
            return super().label(decimals=decimals, base_label=base_label, cache=cache)
        basis_string = ''.join((str(int(i)) for i in self.parameters[0]))
        return f'|{basis_string}⟩⟨{basis_string}|'