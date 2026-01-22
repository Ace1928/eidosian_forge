from functools import wraps
from inspect import signature
import warnings
import numpy as np
from autoray import numpy as anp
import pennylane as qml
def _parse_shifts(shifts, R, arg_name, par_idx, atol, need_f0):
    """Processes shifts for a single reconstruction and determines
    wheter the function at the reconstruction point, ``f0`` will be
    needed.
    """
    _shifts = shifts.get(arg_name)
    if _shifts is not None:
        _shifts = _shifts.get(par_idx)
    if _shifts is not None:
        if len(_shifts) != 2 * R + 1:
            raise ValueError(f'The number of provided shifts ({len(_shifts)}) does not fit to the number of frequencies (2R+1={2 * R + 1}) for parameter {par_idx} in argument {arg_name}.')
        if any(qml.math.isclose(_shifts, qml.math.zeros_like(_shifts), rtol=0, atol=atol)):
            return (_shifts, True)
        return (_shifts, False or need_f0)
    return (_shifts, True)