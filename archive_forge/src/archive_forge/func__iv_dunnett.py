from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._stats_py import _var
def _iv_dunnett(samples: Sequence[npt.ArrayLike], control: npt.ArrayLike, alternative: Literal['two-sided', 'less', 'greater'], random_state: SeedType) -> tuple[list[np.ndarray], np.ndarray, SeedType]:
    """Input validation for Dunnett's test."""
    rng = check_random_state(random_state)
    if alternative not in {'two-sided', 'less', 'greater'}:
        raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")
    ndim_msg = 'Control and samples groups must be 1D arrays'
    n_obs_msg = 'Control and samples groups must have at least 1 observation'
    control = np.asarray(control)
    samples_ = [np.asarray(sample) for sample in samples]
    samples_control: list[np.ndarray] = samples_ + [control]
    for sample in samples_control:
        if sample.ndim > 1:
            raise ValueError(ndim_msg)
        if sample.size < 1:
            raise ValueError(n_obs_msg)
    return (samples_, control, rng)