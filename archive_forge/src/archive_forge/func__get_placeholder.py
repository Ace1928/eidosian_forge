from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
def _get_placeholder(config: Any, prefix: Tuple, path: Tuple):
    if not path:
        return (prefix, config)
    key = path[0]
    if isinstance(config, tuple):
        if config[0] in (_FunctionResolver.TOKEN, _RefResolver.TOKEN):
            return (prefix, config)
        elif key < len(config):
            return _get_placeholder(config[key], prefix=prefix + (path[0],), path=path[1:])
    elif isinstance(config, dict) and key in config or (isinstance(config, list) and key < len(config)):
        return _get_placeholder(config[key], prefix=prefix + (path[0],), path=path[1:])
    return (None, None)