import warnings
from pennylane import math as np
from pennylane.operation import AnyWires, Channel
class DepolarizingChannel(Channel):
    """
    Single-qubit symmetrically depolarizing error channel.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \\sqrt{1-p} \\begin{bmatrix}
                1 & 0 \\\\
                0 & 1
                \\end{bmatrix}

    .. math::
        K_1 = \\sqrt{p/3}\\begin{bmatrix}
                0 & 1  \\\\
                1 & 0
                \\end{bmatrix}

    .. math::
        K_2 = \\sqrt{p/3}\\begin{bmatrix}
                0 & -i \\\\
                i & 0
                \\end{bmatrix}

    .. math::
        K_3 = \\sqrt{p/3}\\begin{bmatrix}
                1 & 0 \\\\
                0 & -1
                \\end{bmatrix}

    where :math:`p \\in [0, 1]` is the depolarization probability and is equally
    divided in the application of all Pauli operations.

    .. note::

        Multiple equivalent definitions of the Kraus operators :math:`\\{K_0 \\ldots K_3\\}` exist in
        the literature [`1 <https://michaelnielsen.org/qcqi/>`_] (Eqs. 8.102-103). Here, we adopt the
        one from Eq. 8.103, which is also presented in [`2 <http://theory.caltech.edu/~preskill/ph219/chap3_15.pdf>`_] (Eq. 3.85).
        For this definition, please make a note of the following:

        * For :math:`p = 0`, the channel will be an Identity channel, i.e., a noise-free channel.
        * For :math:`p = \\frac{3}{4}`, the channel will be a fully depolarizing channel.
        * For :math:`p = 1`, the channel will be a uniform Pauli error channel.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        p (float): Each Pauli gate is applied with probability :math:`\\frac{p}{3}`
        wires (Sequence[int] or int): the wire the channel acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = 'A'
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):
        """Kraus matrices representing the depolarizing channel.

        Args:
            p (float): each Pauli gate is applied with probability :math:`\\frac{p}{3}`

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.DepolarizingChannel.compute_kraus_matrices(0.5)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[0.        , 0.40824829], [0.40824829, 0.        ]]),
         array([[0.+0.j        , 0.-0.40824829j], [0.+0.40824829j, 0.+0.j        ]]),
         array([[ 0.40824829,  0.        ], [ 0.        , -0.40824829]])]
        """
        if not np.is_abstract(p) and (not 0.0 <= p <= 1.0):
            raise ValueError('p must be in the interval [0,1]')
        if np.get_interface(p) == 'tensorflow':
            p = np.cast_like(p, 1j)
        K0 = np.sqrt(1 - p + np.eps) * np.convert_like(np.eye(2, dtype=complex), p)
        K1 = np.sqrt(p / 3 + np.eps) * np.convert_like(np.array([[0, 1], [1, 0]], dtype=complex), p)
        K2 = np.sqrt(p / 3 + np.eps) * np.convert_like(np.array([[0, -1j], [1j, 0]], dtype=complex), p)
        K3 = np.sqrt(p / 3 + np.eps) * np.convert_like(np.array([[1, 0], [0, -1]], dtype=complex), p)
        return [K0, K1, K2, K3]