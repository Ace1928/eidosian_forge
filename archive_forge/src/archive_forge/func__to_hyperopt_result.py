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
def _to_hyperopt_result(self, result: Dict) -> Dict:
    try:
        return {'loss': self.metric_op * result[self.metric], 'status': 'ok'}
    except KeyError as e:
        raise RuntimeError(f'Hyperopt expected to see the metric `{self.metric}` in the last result, but it was not found. To fix this, make sure your call to `tune.report` or your return value of your trainable class `step()` contains the above metric as a key.') from e