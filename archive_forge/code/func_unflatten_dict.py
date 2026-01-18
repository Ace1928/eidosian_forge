import copy
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, TypeVar, Union
from ray.util.annotations import Deprecated
@Deprecated
def unflatten_dict(dt: Dict[str, T], delimiter: str='/') -> Dict[str, T]:
    """Unflatten dict. Does not support unflattening lists."""
    dict_type = type(dt)
    out = dict_type()
    for key, val in dt.items():
        path = key.split(delimiter)
        item = out
        for k in path[:-1]:
            item = item.setdefault(k, dict_type())
            if not isinstance(item, dict_type):
                raise TypeError(f"Cannot unflatten dict due the key '{key}' having a parent key '{k}', which value is not of type {dict_type} (got {type(item)}). Change the key names to resolve the conflict.")
        item[path[-1]] = val
    return out