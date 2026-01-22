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
def _load_policy(self, policy_file: str) -> Tuple[Dict, List[Tuple[int, Dict]]]:
    raw_policy = []
    with open(policy_file, 'rt') as fp:
        for row in fp.readlines():
            try:
                parsed_row = json.loads(row)
            except json.JSONDecodeError:
                raise ValueError('Could not read PBT policy file: {}.'.format(policy_file)) from None
            raw_policy.append(tuple(parsed_row))
    policy = []
    last_new_tag = None
    last_old_conf = None
    for old_tag, new_tag, old_step, new_step, old_conf, new_conf in reversed(raw_policy):
        if last_new_tag and old_tag != last_new_tag:
            break
        last_new_tag = new_tag
        last_old_conf = old_conf
        policy.append((new_step, new_conf))
    return (last_old_conf, list(reversed(policy)))