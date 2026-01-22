from collections import defaultdict
import logging
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import Domain, Float, Quantized, Uniform
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import is_nan_or_inf, unflatten_dict
from ray.tune.utils import flatten_dict
def _setup_optimizer(self):
    if self._metric is None and self._mode:
        self._metric = DEFAULT_METRIC
    self.optimizer = byo.BayesianOptimization(f=None, pbounds=self._space, verbose=self._verbose, random_state=self._random_state)
    if self._analysis is not None:
        self.register_analysis(self._analysis)