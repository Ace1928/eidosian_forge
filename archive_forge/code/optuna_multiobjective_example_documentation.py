import time
import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
This example demonstrates the usage of Optuna with Ray Tune for
multi-objective optimization.

Please note that schedulers may not work correctly with multi-objective
optimization.

Requires the Optuna library to be installed (`pip install optuna`).
