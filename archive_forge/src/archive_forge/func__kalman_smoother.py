import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
@property
def _kalman_smoother(self):
    prefix = self.prefix
    if prefix in self._kalman_smoothers:
        return self._kalman_smoothers[prefix]
    return None