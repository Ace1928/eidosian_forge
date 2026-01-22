import warnings
import inspect
import logging
import semantic_version
import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from . import DefaultQubitLegacy
@staticmethod
def _conj(array):
    if isinstance(array, torch.Tensor):
        return torch.conj(array)
    return np.conj(array)