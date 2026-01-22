import functools
import itertools
import numbers
import warnings
import numpy as np
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.functions import bind_new_parameters
from pennylane.tape import QuantumScript
@functools.lru_cache(maxsize=None)
def _get_shift_rule(frequencies, shifts=None):
    n_freqs = len(frequencies)
    frequencies = qml.math.sort(qml.math.stack(frequencies))
    freq_min = frequencies[0]
    if len(set(frequencies)) != n_freqs or freq_min <= 0:
        raise ValueError(f'Expected frequencies to be a list of unique positive values, instead got {frequencies}.')
    mu = np.arange(1, n_freqs + 1)
    if shifts is None:
        shifts = (2 * mu - 1) * np.pi / (2 * n_freqs * freq_min)
        equ_shifts = True
    else:
        shifts = qml.math.sort(qml.math.stack(shifts))
        if len(shifts) != n_freqs:
            raise ValueError(f'Expected number of shifts to equal the number of frequencies ({n_freqs}), instead got {shifts}.')
        if len(set(shifts)) != n_freqs:
            raise ValueError(f'Shift values must be unique, instead got {shifts}')
        equ_shifts = np.allclose(shifts, (2 * mu - 1) * np.pi / (2 * n_freqs * freq_min))
    if len(set(np.round(np.diff(frequencies), 10))) <= 1 and equ_shifts:
        coeffs = freq_min * (-1) ** (mu - 1) / (4 * n_freqs * np.sin(np.pi * (2 * mu - 1) / (4 * n_freqs)) ** 2)
    else:
        sin_matrix = -4 * np.sin(np.outer(shifts, frequencies))
        det_sin_matrix = np.linalg.det(sin_matrix)
        if abs(det_sin_matrix) < 1e-06:
            warnings.warn(f'Solving linear problem with near zero determinant ({det_sin_matrix}) may give unstable results for the parameter shift rules.')
        coeffs = -2 * linalg_solve(sin_matrix.T, frequencies)
    coeffs = np.concatenate((coeffs, -coeffs))
    shifts = np.concatenate((shifts, -shifts))
    return np.stack([coeffs, shifts]).T