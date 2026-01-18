import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
@property
def time_invariant(self):
    """
        (bool) Whether or not currently active representation matrices are
        time-invariant
        """
    if self._time_invariant is None:
        return self._design.shape[2] == self._obs_intercept.shape[1] == self._obs_cov.shape[2] == self._transition.shape[2] == self._state_intercept.shape[1] == self._selection.shape[2] == self._state_cov.shape[2]
    else:
        return self._time_invariant