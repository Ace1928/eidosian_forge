import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_bandit_linucb():
    from ray.rllib.algorithms.bandit.bandit import BanditLinUCB
    return (BanditLinUCB, BanditLinUCB.get_default_config())