from __future__ import annotations
import functools
import warnings
from collections.abc import Mapping, Callable
from copy import deepcopy
from typing import Any
import numpy as np
import symengine as sym
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
def _lifted_gaussian(t: sym.Symbol, center: sym.Symbol | sym.Expr | complex, t_zero: sym.Symbol | sym.Expr | complex, sigma: sym.Symbol | sym.Expr | complex) -> sym.Expr:
    """Helper function that returns a lifted Gaussian symbolic equation.

    For :math:`\\sigma=` ``sigma`` the symbolic equation will be

    .. math::

        f(x) = \\exp\\left(-\\frac12 \\left(\\frac{x - \\mu}{\\sigma}\\right)^2 \\right),

    with the center :math:`\\mu=` ``duration/2``.
    Then, each output sample :math:`y` is modified according to:

    .. math::

        y \\mapsto \\frac{y-y^*}{1.0-y^*},

    where :math:`y^*` is the value of the un-normalized Gaussian at the endpoints of the pulse.
    This sets the endpoints to :math:`0` while preserving the amplitude at the center,
    i.e. :math:`y` is set to :math:`1.0`.

    Args:
        t: Symbol object representing time.
        center: Symbol or expression representing the middle point of the samples.
        t_zero: The value of t at which the pulse is lowered to 0.
        sigma: Symbol or expression representing Gaussian sigma.

    Returns:
        Symbolic equation.
    """
    t_shifted = (t - center).expand()
    t_offset = (t_zero - center).expand()
    gauss = sym.exp(-(t_shifted / sigma) ** 2 / 2)
    offset = sym.exp(-(t_offset / sigma) ** 2 / 2)
    return (gauss - offset) / (1 - offset)