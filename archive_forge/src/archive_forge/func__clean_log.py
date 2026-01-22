import enum
import os
import pickle
import urllib
import warnings
import numpy as np
from numbers import Number
import pyarrow.fs
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ray
from ray import logger
from ray.air import session
from ray.air._internal import usage as air_usage
from ray.air.util.node import _force_on_current_node
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.tune.experiment import Trial
from ray.train._internal.syncer import DEFAULT_SYNC_TIMEOUT
from ray._private.storage import _load_class
from ray.util import PublicAPI
from ray.util.queue import Queue
def _clean_log(obj: Any):
    if isinstance(obj, dict):
        return {k: _clean_log(v) for k, v in obj.items()}
    elif isinstance(obj, (list, set)):
        return [_clean_log(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple((_clean_log(v) for v in obj))
    elif _is_allowed_type(obj):
        return obj
    try:
        json_dumps_safer(obj)
        pickle.dumps(obj)
        return obj
    except Exception:
        fallback = str(obj)
        try:
            fallback = int(fallback)
            return fallback
        except ValueError:
            pass
        try:
            fallback = float(fallback)
            return fallback
        except ValueError:
            pass
        return fallback