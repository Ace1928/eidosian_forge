import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_alpha_zero():
    import ray.rllib.algorithms.alpha_zero as alpha_zero
    return (alpha_zero.AlphaZero, alpha_zero.AlphaZero.get_default_config())