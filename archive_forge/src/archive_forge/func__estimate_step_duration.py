import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import threading
import time
import types as python_types
import warnings
import weakref
import numpy as np
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def _estimate_step_duration(self, current, now):
    """Estimate the duration of a single step.

    Given the step number `current` and the corresponding time `now`
    this function returns an estimate for how long a single step
    takes. If this is called before one step has been completed
    (i.e. `current == 0`) then zero is given as an estimate. The duration
    estimate ignores the duration of the (assumed to be non-representative)
    first step for estimates when more steps are available (i.e. `current>1`).
    Args:
      current: Index of current step.
      now: The current time.
    Returns: Estimate of the duration of a single step.
    """
    if current:
        if self._time_after_first_step is not None and current > 1:
            time_per_unit = (now - self._time_after_first_step) / (current - 1)
        else:
            time_per_unit = (now - self._start) / current
        if current == 1:
            self._time_after_first_step = now
        return time_per_unit
    else:
        return 0