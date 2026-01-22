import warnings
import inspect
import logging
import semantic_version
import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from . import DefaultQubitLegacy
@staticmethod
def _dot(x, y):
    if x.device != y.device:
        if x.device != 'cpu':
            return torch.tensordot(x, y.to(x.device), dims=1)
        if y.device != 'cpu':
            return torch.tensordot(x.to(y.device), y, dims=1)
    return torch.tensordot(x, y, dims=1)