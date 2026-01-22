import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
def _pack_args(self_fn_args: Iterable[Any], self_fn_kwargs: Dict[str, Any], prev_fn_args: Iterable[Any], prev_fn_kwargs: Dict[str, Any]) -> Tuple[Tuple[Any], Callable[[Tuple[Any]], Tuple[Tuple[Any], Dict[str, Any], Tuple[Any], Dict[str, Any]]]]:
    """Pack the (kw)args from two stages into a single, flat positional args tuple that
    can be given to a Ray task, ensuring resoultion of each argument.
    This function returns this args tuple along with a function that will unpack this
    flat args tuple back into it's original args and kwargs structure.
    """
    if not self_fn_args:
        self_fn_args = tuple()
    if not self_fn_kwargs:
        self_fn_kwargs = {}
    if not prev_fn_args:
        prev_fn_args = tuple()
    if not prev_fn_kwargs:
        prev_fn_kwargs = {}
    offsets = list(itertools.accumulate([len(self_fn_args), len(prev_fn_args), len(self_fn_kwargs), len(prev_fn_kwargs)]))
    keys = list(self_fn_kwargs.keys()) + list(prev_fn_kwargs.keys())
    fn_args = self_fn_args + prev_fn_args + tuple(self_fn_kwargs.values()) + tuple(prev_fn_kwargs.values())

    def unpack(fn_args: List[Any]) -> Tuple[List[Any], Dict[str, Any], List[Any], Dict[str, Any]]:
        self_fn_args = fn_args[:offsets[0]]
        prev_fn_args = fn_args[offsets[0]:offsets[1]]
        self_fn_kwargs = fn_args[offsets[1]:offsets[2]]
        prev_fn_kwargs = fn_args[offsets[2]:]
        prev_key_offset = offsets[2] - offsets[1]
        self_fn_kwargs = {k: v for k, v in zip(keys[:prev_key_offset], self_fn_kwargs)}
        prev_fn_kwargs = {k: v for k, v in zip(keys[prev_key_offset:], prev_fn_kwargs)}
        return (self_fn_args, self_fn_kwargs, prev_fn_args, prev_fn_kwargs)
    return (fn_args, unpack)