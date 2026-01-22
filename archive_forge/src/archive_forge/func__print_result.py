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
def _print_result(self, trial, result: Optional[Dict]=None, force: bool=False):
    """Only print result if a different result has been reported, or force=True"""
    result = result or trial.last_result
    last_result_iter = self._trial_last_printed_results.get(trial.trial_id, -1)
    this_iter = result.get(TRAINING_ITERATION, 0)
    if this_iter != last_result_iter or force:
        _print_dict_as_table(result, header=f'{self._addressing_tmpl.format(trial)} result', include=self._progress_metrics, exclude=BLACKLISTED_KEYS, division=AUTO_RESULT_KEYS)
        self._trial_last_printed_results[trial.trial_id] = this_iter