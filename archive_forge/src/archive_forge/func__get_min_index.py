import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
def _get_min_index(self, values: List[int], disallowed_indices: Optional[Set[int]]=None) -> int:
    """
        Return ``values.index(min(values))``, except only uses one pass.

        It also excludes any indices in ``disallowed_indices`` if provided.

        Arguments:
            values: (List[int]): :class:`list` of values.
            disallowed_indices (Optional[Set[int]]): indices that are
                disallowed from being the returned min index.
        """
    min_index = -1
    min_value = float('inf')
    for i, value in enumerate(values):
        if disallowed_indices and i in disallowed_indices:
            continue
        if value < min_value:
            min_value = value
            min_index = i
    assert min_index >= 0, 'All indices are disallowed'
    return min_index