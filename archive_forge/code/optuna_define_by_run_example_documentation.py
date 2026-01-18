import time
from typing import Dict, Optional, Any
import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
Define-by-run function to create the search space.

    Ensure no actual computation takes place here. That should go into
    the trainable passed to ``Tuner`` (in this example, that's
    ``easy_objective``).

    For more information, see https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html

    This function should either return None or a dict with constant values.
    