import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_td3():
    import ray.rllib.algorithms.td3 as td3
    return (td3.TD3, td3.TD3.get_default_config())