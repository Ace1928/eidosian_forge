import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
def _is_numba_core_config_loaded():
    """
    To detect if numba.core.config has been initialized due to circular imports.
    """
    try:
        numba.core.config
    except AttributeError:
        return False
    else:
        return True