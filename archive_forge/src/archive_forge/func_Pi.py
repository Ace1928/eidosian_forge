import json
from os import path
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
def Pi(j):
    """Projector on eigenspace of eigenvalue E_i"""
    return np.outer(np.conjugate(eigvecs[:, j]), eigvecs[:, j])