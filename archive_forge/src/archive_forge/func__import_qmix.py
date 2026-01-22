import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_qmix():
    import ray.rllib.algorithms.qmix as qmix
    return (qmix.QMix, qmix.QMix.get_default_config())