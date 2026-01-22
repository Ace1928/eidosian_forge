import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_dt():
    import ray.rllib.algorithms.dt as dt
    return (dt.DT, dt.DT.get_default_config())