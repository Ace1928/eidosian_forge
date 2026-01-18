import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def thermal_state(nbar, hbar=2.0):
    """Returns a thermal state.

    Args:
        nbar (float): the mean photon number
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`

    Returns:
        array: the thermal state
    """
    means = np.zeros([2])
    state = [(2 * nbar + 1) * np.identity(2) * hbar / 2, means]
    return state