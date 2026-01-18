import collections
import logging
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util.annotations import PublicAPI
Callback for early stopping.

        This stopping rule stops a running trial if the trial's best objective
        value by step `t` is strictly worse than the median of the running
        averages of all completed trials' objectives reported up to step `t`.
        