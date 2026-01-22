import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def _cur_band_filled(self) -> bool:
    """Checks if the current band is filled.

        The size of the current band should be equal to s_max_1"""
    cur_band = self._hyperbands[self._state['band_idx']]
    return len(cur_band) == self._s_max_1