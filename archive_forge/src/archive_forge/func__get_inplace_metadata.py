import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple
import torch
from torch.testing._internal.composite_compliance import (
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def _get_inplace_metadata(self, func, out) -> Tuple[int, int, Tuple[int, ...]]:
    curr_idx = len(self.data)

    def get_tensor_id(e):
        return e.untyped_storage().data_ptr() if isinstance(e, torch.Tensor) else None
    output_ids = tree_map(get_tensor_id, out)
    if not is_inplace(func):
        return (curr_idx, output_ids, ())
    op_id = curr_idx
    op_parent_id = -1
    for i, d in enumerate(self.data):
        past_output_ids = d.output_ids
        past_output_ids = [past_output_ids] if not isinstance(past_output_ids, (list, tuple, dict)) else past_output_ids
        if output_ids in past_output_ids:
            op_parent_id = i
            break
    if op_parent_id < 0:
        op_parent_id = op_id
    inplace_info = (op_id, op_parent_id)
    return (curr_idx, output_ids, inplace_info)