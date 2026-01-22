import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_impala():
    import ray.rllib.algorithms.impala as impala
    return (impala.Impala, impala.Impala.get_default_config())