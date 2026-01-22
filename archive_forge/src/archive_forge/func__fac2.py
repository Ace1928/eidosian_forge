import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _fac2(n):
    """Compute the double factorial of an integer.

    The function uses the definition :math:`(-1)!! = 1`.

    Args:
        n (int): number for which the double factorial is computed

    Returns:
        int: the computed double factorial

    """
    return int(fac2(n) if n != -1 else 1)