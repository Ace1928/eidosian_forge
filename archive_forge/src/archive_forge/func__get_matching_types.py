import itertools
from typing import Any, List, OrderedDict, Set, Optional, Callable
import operator
from torch.fx import Node
import torch
from torch.fx.passes.utils.source_matcher_utils import (
def _get_matching_types(partition_type):
    matching_types = [partition_type]
    if partition_type in _EQUIVALENT_TYPES_DICT:
        matching_types.extend(_EQUIVALENT_TYPES_DICT[partition_type])
    return matching_types