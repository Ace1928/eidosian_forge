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
Notification for the completion of trial.

        The result is internally negated when interacting with HyperOpt
        so that HyperOpt can "maximize" this value, as it minimizes on default.
        