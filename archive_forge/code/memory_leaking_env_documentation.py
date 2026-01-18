import logging
import uuid
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils.annotations import override
An env that leaks very little memory.

    Useful for proving that our memory-leak tests can catch the
    slightest leaks.
    