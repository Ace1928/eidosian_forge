import cmath
from typing import Tuple
import numpy as np
def BARENCO(alpha: float, phi: float, theta: float) -> np.ndarray:
    lower_unitary = np.array([[np.exp(1j * phi) * np.cos(theta), -1j * np.exp(1j * (alpha - phi)) * np.sin(theta)], [-1j * np.exp(1j * (alpha + phi)) * np.sin(theta), np.exp(1j * alpha) * np.cos(theta)]])
    return np.kron(P0, np.eye(2)) + np.kron(P1, lower_unitary)