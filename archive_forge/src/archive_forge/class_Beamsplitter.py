import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class Beamsplitter(CVOperation):
    """
    Beamsplitter interaction.

    .. math::
        B(\\theta,\\phi) = \\exp\\left(\\theta (e^{i \\phi} \\a \\hat{b}^\\dagger -e^{-i \\phi}\\ad \\hat{b}) \\right).

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2
    * Gradient recipe: :math:`\\frac{d}{d \\theta}f(B(\\theta,\\phi)) = \\frac{1}{2} \\left[f(B(\\theta+\\pi/2, \\phi)) - f(B(\\theta-\\pi/2, \\phi))\\right]`
      where :math:`f` is an expectation value depending on :math:`B(\\theta,\\phi)`.
    * Heisenberg representation:

      .. math:: M = \\begin{bmatrix}
            1 & 0 & 0 & 0 & 0\\\\
            0 & \\cos\\theta & 0 & -\\cos\\phi\\sin\\theta & -\\sin\\phi\\sin\\theta \\\\
            0 & 0 & \\cos\\theta & \\sin\\phi\\sin\\theta & -\\cos\\phi\\sin\\theta\\\\
            0 & \\cos\\phi\\sin\\theta & -\\sin\\phi\\sin\\theta & \\cos\\theta & 0\\\\
            0 & \\sin\\phi\\sin\\theta & \\cos\\phi\\sin\\theta & 0 & \\cos\\theta
        \\end{bmatrix}

    Args:
        theta (float): Transmittivity angle :math:`\\theta`. The transmission amplitude
            of the beamsplitter is :math:`t = \\cos(\\theta)`.
            The value :math:`\\theta=\\pi/4` gives the 50-50 beamsplitter.
        phi (float): Phase angle :math:`\\phi`. The reflection amplitude of the
            beamsplitter is :math:`r = e^{i\\phi}\\sin(\\theta)`.
            The value :math:`\\phi = \\pi/2` gives the symmetric beamsplitter.
        wires (Sequence[Any]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 2
    num_wires = 2
    grad_method = 'A'
    grad_recipe = (_two_term_shift_rule, _two_term_shift_rule)

    def __init__(self, theta, phi, wires, id=None):
        super().__init__(theta, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1], bare=True)
        c = math.cos(p[0])
        s = math.sin(p[0])
        U = c * np.eye(5)
        U[0, 0] = 1
        U[1:3, 3:5] = -s * R.T
        U[3:5, 1:3] = s * R
        return U

    def adjoint(self):
        theta, phi = self.parameters
        return Beamsplitter(-theta, phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'BS', cache=cache)