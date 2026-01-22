import collections
import logging
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util.annotations import PublicAPI
def _best_result(self, trial):
    results = self._results[trial]
    return self._compare_op([r[self._metric] for r in results])