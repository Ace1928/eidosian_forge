import datetime
from typing import Union
import time
from ray import logger
from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper
Stops all trials after a certain timeout.

    This stopper is automatically created when the `time_budget_s`
    argument is passed to `train.RunConfig()`.

    Args:
        timeout: Either a number specifying the timeout in seconds, or
            a `datetime.timedelta` object.
    