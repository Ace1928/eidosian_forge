import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_ddppo():
    import ray.rllib.algorithms.ddppo as ddppo
    return (ddppo.DDPPO, ddppo.DDPPO.get_default_config())