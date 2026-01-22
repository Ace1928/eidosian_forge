import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable
def _get_or(self, key: Union[int, str], expected_type: type, throw: bool=True) -> Any:
    if isinstance(key, str) and key in self or isinstance(key, int):
        return as_type(self[key], expected_type)
    if throw:
        raise KeyError(f'{key} not found')
    return None