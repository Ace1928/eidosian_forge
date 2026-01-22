import argparse
import sys
from typing import (
import collections
from dataclasses import dataclass
import datetime
from enum import IntEnum
import logging
import math
import numbers
import numpy as np
import os
import pandas as pd
import textwrap
import time
from ray.air._internal.usage import AirEntrypoint
from ray.train import Checkpoint
from ray.tune.search.sample import Domain
from ray.tune.utils.log import Verbosity
import ray
from ray._private.dict import unflattened_lookup, flatten_dict
from ray._private.thirdparty.tabulate.tabulate import (
from ray.air.constants import TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.result import (
from ray.tune.experiment.trial import Trial
def _get_trial_info(trial: Trial, param_keys: List[str], metric_keys: List[str]) -> List[str]:
    """Returns the following information about a trial:

    name | status | metrics...

    Args:
        trial: Trial to get information for.
        param_keys: Names of parameters to include.
        metric_keys: Names of metrics to include.
    """
    result = trial.last_result
    trial_info = [str(trial), trial.status]
    trial_info.extend([_max_len(unflattened_lookup(param, trial.config, default=None)) for param in param_keys])
    trial_info.extend([_max_len(unflattened_lookup(metric, result, default=None)) for metric in metric_keys])
    return trial_info