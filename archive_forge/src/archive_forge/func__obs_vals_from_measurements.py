import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
def _obs_vals_from_measurements(bitstrings: np.ndarray, qubit_to_index: Dict['cirq.Qid', int], observable: 'cirq.PauliString', atol: float):
    """Multiply together bitstrings to get observed values of operators."""
    idxs = [qubit_to_index[q] for q in observable.keys()]
    selected_bitstrings = np.asarray(bitstrings[:, idxs], dtype=np.int8)
    selected_obsstrings = 1 - 2 * selected_bitstrings
    coeff = _check_and_get_real_coef(observable, atol=atol)
    obs_vals = coeff * np.prod(selected_obsstrings, axis=1)
    return obs_vals