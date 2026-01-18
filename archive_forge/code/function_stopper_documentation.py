from typing import Dict, Callable
from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper
Provide a custom function to check if trial should be stopped.

    The passed function will be called after each iteration. If it returns
    True, the trial will be stopped.

    Args:
        function: Function that checks if a trial
            should be stopped. Must accept the `trial_id` string  and `result`
            dictionary as arguments. Must return a boolean.

    