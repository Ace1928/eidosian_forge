import copy
import logging
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils import flatten_dict
from ray.tune.utils.util import is_nan_or_inf, unflatten_dict, validate_warmstart
Notification for the completion of trial.

        The result is internally negated when interacting with Skopt
        so that Skopt Optimizers can "maximize" this value,
        as it minimizes on default.
        