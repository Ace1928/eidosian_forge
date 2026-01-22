from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from triad import SerializableRLock
from tune._utils.math import adjust_high
from tune.concepts.flow import Trial, TrialReport
from tune.concepts.logger import make_logger
from tune.concepts.space import (
from tune.noniterative.objective import (
def _convert_rand(k: str, v: Rand) -> Any:
    if v.q is None:
        if not v.log:
            return (hp.uniform(k, v.low, v.high), lambda x: x)
        else:
            return (hp.loguniform(k, np.log(v.low), np.log(v.high)), lambda x: x)
    else:
        _high = adjust_high(0, v.high - v.low, v.q, include_high=v.include_high)
        n = int(np.round(_high / v.q))
        if not v.log:
            return (hp.randint(k, 0, n) * v.q + v.low, lambda x: x)
        else:
            _high = max(0, _high - v.q)
            return (hp.qloguniform(k, 0, np.log(_high), q=v.q) + v.low, lambda x: x)