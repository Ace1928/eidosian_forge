import warnings
import numpy as np
from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
def _get_scalar(self, var):
    """Get scalar value from HDF5 scalar"""
    return var[()]