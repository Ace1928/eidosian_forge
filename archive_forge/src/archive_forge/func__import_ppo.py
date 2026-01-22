import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_ppo():
    import ray.rllib.algorithms.ppo as ppo
    return (ppo.PPO, ppo.PPO.get_default_config())