import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
def _safe_set_output(estimator, *, transform=None):
    """Safely call estimator.set_output and error if it not available.

    This is used by meta-estimators to set the output for child estimators.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance.

    transform : {"default", "pandas"}, default=None
        Configure output of the following estimator's methods:

        - `"transform"`
        - `"fit_transform"`

        If `None`, this operation is a no-op.

    Returns
    -------
    estimator : estimator instance
        Estimator instance.
    """
    set_output_for_transform = hasattr(estimator, 'transform') or (hasattr(estimator, 'fit_transform') and transform is not None)
    if not set_output_for_transform:
        return
    if not hasattr(estimator, 'set_output'):
        raise ValueError(f'Unable to configure output for {estimator} because `set_output` is not available.')
    return estimator.set_output(transform=transform)