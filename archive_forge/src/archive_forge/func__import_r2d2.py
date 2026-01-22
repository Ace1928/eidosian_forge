import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_r2d2():
    import ray.rllib.algorithms.r2d2 as r2d2
    return (r2d2.R2D2, r2d2.R2D2.get_default_config())