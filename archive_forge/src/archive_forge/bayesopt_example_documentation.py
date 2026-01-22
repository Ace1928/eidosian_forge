import time
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
This example demonstrates the usage of BayesOpt with Ray Tune.

It also checks that it is usable with a separate scheduler.

Requires the BayesOpt library to be installed (`pip install bayesian-optimization`).
