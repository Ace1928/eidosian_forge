import copy
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from ray.air.constants import TRAINING_ITERATION
from ray.train import Checkpoint
from ray.train._internal.session import _TrainingResult, _FutureTrainingResult
from ray.tune.error import TuneError
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search import SearchGenerator
from ray.tune.utils.util import SafeFallbackEncoder
from ray.tune.search.sample import Domain, Function
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.search.variant_generator import format_vars
from ray.tune.experiment import Trial
from ray.util import PublicAPI
from ray.util.debug import log_once
def _quantiles(self) -> Tuple[List[Trial], List[Trial]]:
    """Returns trials in the lower and upper `quantile` of the population.

        If there is not enough data to compute this, returns empty lists.
        """
    trials = []
    for trial, state in self._trial_state.items():
        logger.debug('Trial {}, state {}'.format(trial, state))
        if trial.is_finished():
            logger.debug('Trial {} is finished'.format(trial))
        if state.last_score is not None and (not trial.is_finished()):
            trials.append(trial)
    trials.sort(key=lambda t: self._trial_state[t].last_score)
    if len(trials) <= 1:
        return ([], [])
    else:
        num_trials_in_quantile = int(math.ceil(len(trials) * self._quantile_fraction))
        if num_trials_in_quantile > len(trials) / 2:
            num_trials_in_quantile = int(math.floor(len(trials) / 2))
        return (trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:])