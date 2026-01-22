import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
@functools.lru_cache(None)
def _sfdp_init():
    from .serialized_patterns.central_index import get_serialized_pattern
    for key, register_replacement_kwargs in _get_sfdp_patterns():
        search_fn_pattern = get_serialized_pattern(key)
        register_replacement(**register_replacement_kwargs, search_fn_pattern=search_fn_pattern)