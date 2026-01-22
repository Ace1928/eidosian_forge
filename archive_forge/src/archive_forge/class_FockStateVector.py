import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class FockStateVector(CVOperation):
    """
    Prepare subsystems using the given ket vector in the Fock basis.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        state (array): a single ket vector, for single mode state preparation,
            or a multimode ket, with one array dimension per mode
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    .. details::
        :title: Usage Details

        For a single mode with cutoff dimension :math:`N`, the input is a
        1-dimensional vector of length :math:`N`.

        .. code-block::

            dev_fock = qml.device("strawberryfields.fock", wires=4, cutoff_dim=4)

            state = np.array([0, 0, 1, 0])

            @qml.qnode(dev_fock)
            def circuit():
                qml.FockStateVector(state, wires=0)
                return qml.expval(qml.NumberOperator(wires=0))

        For multiple modes, the input is the tensor product of single mode
        kets. For example, given a set of :math:`M` single mode vectors of
        length :math:`N`, the input should have shape ``(N, ) * M``.

        .. code-block::

            used_wires = [0, 3]
            cutoff_dim = 5

            dev_fock = qml.device("strawberryfields.fock", wires=4, cutoff_dim=cutoff_dim)

            state_1 = np.array([0, 1, 0, 0, 0])
            state_2 = np.array([0, 0, 0, 1, 0])

            combined_state = np.kron(state_1, state_2).reshape(
                (cutoff_dim, ) * len(used_wires)
            )

            @qml.qnode(dev_fock)
            def circuit():
                qml.FockStateVector(combined_state, wires=used_wires)
                return qml.expval(qml.NumberOperator(wires=0))

    """
    num_params = 1
    num_wires = AnyWires
    grad_method = 'F'

    def __init__(self, state, wires, id=None):
        super().__init__(state, wires=wires, id=id)

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

        >>> qml.FockStateVector([1,2,3], wires=(0,1,2)).label()
        '|123⟩'

        """
        if base_label is not None:
            return base_label
        basis_string = ''.join((str(int(i)) for i in self.parameters[0]))
        return f'|{basis_string}⟩'