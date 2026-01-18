from typing import Dict, Optional
from collections import defaultdict, deque
import numpy as np
from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper
Early stop single trials when they reached a plateau.

    When the standard deviation of the `metric` result of a trial is
    below a threshold `std`, the trial plateaued and will be stopped
    early.

    Args:
        metric: Metric to check for convergence.
        std: Maximum metric standard deviation to decide if a
            trial plateaued. Defaults to 0.01.
        num_results: Number of results to consider for stdev
            calculation.
        grace_period: Minimum number of timesteps before a trial
            can be early stopped
        metric_threshold (Optional[float]):
            Minimum or maximum value the result has to exceed before it can
            be stopped early.
        mode: If a `metric_threshold` argument has been
            passed, this must be one of [min, max]. Specifies if we optimize
            for a large metric (max) or a small metric (min). If max, the
            `metric_threshold` has to be exceeded, if min the value has to
            be lower than `metric_threshold` in order to early stop.
    