import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_a2c():
    import ray.rllib.algorithms.a2c as a2c
    return (a2c.A2C, a2c.A2C.get_default_config())