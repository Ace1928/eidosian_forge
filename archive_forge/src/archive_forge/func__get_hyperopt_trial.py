from typing import Any, Dict, List, Optional
import numpy as np
import copy
import logging
from functools import partial
from ray import cloudpickle
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import assign_value, parse_spec_vars
from ray.tune.utils import flatten_dict
from ray.tune.error import TuneError
def _get_hyperopt_trial(self, trial_id: str) -> Optional[Dict]:
    if trial_id not in self._live_trial_mapping:
        return
    hyperopt_tid = self._live_trial_mapping[trial_id][0]
    return [t for t in self._hpopt_trials.trials if t['tid'] == hyperopt_tid][0]