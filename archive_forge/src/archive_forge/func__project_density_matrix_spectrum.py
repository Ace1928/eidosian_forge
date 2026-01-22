import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
def _project_density_matrix_spectrum(rdm):
    """Project the estimator density matrix rdm with possibly negative eigenvalues onto the closest true density matrix in L2 norm"""
    evs = qml.math.eigvalsh(rdm)[::-1]
    d = len(rdm)
    a = 0.0
    for i in range(d - 1, -1, -1):
        if evs[i] + a / (i + 1) > 0:
            break
        a += evs[i]
    lambdas = evs[:i + 1] + a / (i + 1)
    return lambdas[::-1]