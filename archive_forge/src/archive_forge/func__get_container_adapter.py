import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
def _get_container_adapter(method, estimator=None):
    """Get container adapter."""
    dense_config = _get_output_config(method, estimator)['dense']
    try:
        return ADAPTERS_MANAGER.adapters[dense_config]
    except KeyError:
        return None