import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_dreamerv3():
    import ray.rllib.algorithms.dreamerv3 as dreamerv3
    return (dreamerv3.DreamerV3, dreamerv3.DreamerV3.get_default_config())