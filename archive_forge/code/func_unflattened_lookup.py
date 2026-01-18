import copy
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, TypeVar, Union
from ray.util.annotations import Deprecated
@Deprecated
def unflattened_lookup(flat_key: str, lookup: Union[Mapping, Sequence], delimiter: str='/', **kwargs) -> Union[Mapping, Sequence]:
    """
    Unflatten `flat_key` and iteratively look up in `lookup`. E.g.
    `flat_key="a/0/b"` will try to return `lookup["a"][0]["b"]`.
    """
    if flat_key in lookup:
        return lookup[flat_key]
    keys = deque(flat_key.split(delimiter))
    base = lookup
    while keys:
        key = keys.popleft()
        try:
            if isinstance(base, Mapping):
                base = base[key]
            elif isinstance(base, Sequence):
                base = base[int(key)]
            else:
                raise KeyError()
        except KeyError as e:
            if 'default' in kwargs:
                return kwargs['default']
            raise e
    return base