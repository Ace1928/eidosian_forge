import warnings
from pennylane import math as np
from pennylane.operation import AnyWires, Channel
class PhaseDamping(Channel):
    """
    Single-qubit phase damping error channel.

    Interaction with the environment can lead to loss of quantum information changes without any
    changes in qubit excitations. This can be modelled by the phase damping channel, with
    the following Kraus matrices:

    .. math::
        K_0 = \\begin{bmatrix}
                1 & 0 \\\\
                0 & \\sqrt{1-\\gamma}
                \\end{bmatrix}
    .. math::

        K_1 = \\begin{bmatrix}
                0 & 0  \\\\
                0 & \\sqrt{\\gamma}
                \\end{bmatrix}

    where :math:`\\gamma \\in [0, 1]` is the phase damping probability.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        gamma (float): phase damping probability
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 1
    grad_method = 'F'

    def __init__(self, gamma, wires, id=None):
        super().__init__(gamma, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(gamma):
        """Kraus matrices representing the PhaseDamping channel.

        Args:
            gamma (float): phase damping probability

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.PhaseDamping.compute_kraus_matrices(0.5)
        [array([[1.        , 0.        ], [0.        , 0.70710678]]),
         array([[0.        , 0.        ], [0.        , 0.70710678]])]
        """
        if not np.is_abstract(gamma) and (not 0.0 <= gamma <= 1.0):
            raise ValueError('gamma must be in the interval [0,1].')
        K0 = np.diag([1, np.sqrt(1 - gamma + np.eps)])
        K1 = np.diag([0, np.sqrt(gamma + np.eps)])
        return [K0, K1]