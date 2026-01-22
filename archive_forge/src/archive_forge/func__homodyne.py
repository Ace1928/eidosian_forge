import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def _homodyne(cov, mu, params, hbar=2.0):
    """Arbitrary angle homodyne expectation."""
    rot = rotation(phi)
    muphi = rot.T @ mu
    covphi = rot.T @ cov @ rot
    return (muphi[0], covphi[0, 0])