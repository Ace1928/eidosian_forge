import collections
import logging
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util.annotations import PublicAPI
def _on_insufficient_samples(self, tune_controller: 'TuneController', trial: Trial, time: float) -> str:
    pause = time - self._last_pause[trial] > self._min_time_slice
    pause = pause and [t for t in tune_controller.get_live_trials() if t.status in (Trial.PENDING, Trial.PAUSED)]
    return TrialScheduler.PAUSE if pause else TrialScheduler.CONTINUE