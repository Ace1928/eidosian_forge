import time
import logging
import pickle
import functools
import warnings
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict, validate_warmstart
def _suggest_from_define_by_run_func(self, func: Callable[['OptunaTrial'], Optional[Dict[str, Any]]], ot_trial: 'OptunaTrial') -> Dict:
    captor = _OptunaTrialSuggestCaptor(ot_trial)
    time_start = time.time()
    ret = func(captor)
    time_taken = time.time() - time_start
    if time_taken > DEFINE_BY_RUN_WARN_THRESHOLD_S:
        warnings.warn(f"Define-by-run function passed in the `space` argument took {time_taken} seconds to run. Ensure that actual computation, training takes place inside Tune's train functions or Trainables passed to `tune.Tuner()`.")
    if ret is not None:
        if not isinstance(ret, dict):
            raise TypeError(f'The return value of the define-by-run function passed in the `space` argument should be either None or a `dict` with `str` keys. Got {type(ret)}.')
        if not all((isinstance(k, str) for k in ret.keys())):
            raise TypeError('At least one of the keys in the dict returned by the define-by-run function passed in the `space` argument was not a `str`.')
    return {**captor.captured_values, **ret} if ret else captor.captured_values