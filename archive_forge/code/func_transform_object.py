import dataclasses
import io
import logging
import operator
from collections import ChainMap
from functools import reduce
from typing import List, Tuple, Dict, Any, Union, cast
import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.planner import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed.checkpoint.planner_helpers import (
from torch.distributed.checkpoint._nested_dict import (
from torch.distributed.checkpoint._sharded_tensor_utils import (
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.checkpoint._traverse import set_element
def transform_object(self, write_item: WriteItem, object: Any):
    """Extension from the planner interface to make it easy to extend the default planner."""
    if write_item.type == WriteItemType.BYTE_IO:
        bytes = io.BytesIO()
        torch.save(object, bytes)
        object = bytes
    return object