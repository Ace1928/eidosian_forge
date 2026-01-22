from __future__ import print_function
import collections
import datetime
import numbers
import os
import sys
import textwrap
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import ray
from ray._private.dict import flatten_dict
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.experimental.tqdm_ray import safe_print
from ray.air.util.node import _force_on_current_node
from ray.air.constants import EXPR_ERROR_FILE, TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.logger import pretty_print
from ray.tune.result import (
from ray.tune.experiment.trial import DEBUG_PRINT_INTERVAL, Trial, _Location
from ray.tune.trainable import Trainable
from ray.tune.utils import unflattened_lookup
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.queue import Empty, Queue
from ray.widgets import Template
def _fair_filter_trials(trials_by_state: Dict[str, List[Trial]], max_trials: int, sort_by_metric: bool=False):
    """Filters trials such that each state is represented fairly.

    The oldest trials are truncated if necessary.

    Args:
        trials_by_state: Maximum number of trials to return.
    Returns:
        Dict mapping state to List of fairly represented trials.
    """
    num_trials_by_state = collections.defaultdict(int)
    no_change = False
    while max_trials > 0 and (not no_change):
        no_change = True
        for state in sorted(trials_by_state):
            if num_trials_by_state[state] < len(trials_by_state[state]):
                no_change = False
                max_trials -= 1
                num_trials_by_state[state] += 1
    sorted_trials_by_state = dict()
    for state in sorted(trials_by_state):
        if state == Trial.TERMINATED and sort_by_metric:
            sorted_trials_by_state[state] = trials_by_state[state]
        else:
            sorted_trials_by_state[state] = sorted(trials_by_state[state], reverse=False, key=lambda t: t.trial_id)
    filtered_trials = {state: sorted_trials_by_state[state][:num_trials_by_state[state]] for state in sorted(trials_by_state)}
    return filtered_trials