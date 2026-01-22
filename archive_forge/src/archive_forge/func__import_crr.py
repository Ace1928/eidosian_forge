import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_crr():
    from ray.rllib.algorithms import crr
    return (crr.CRR, crr.CRR.get_default_config())