from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING
import torch
from ._internals import (
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func
def _dispatch_custom_op(sharding_spec, op: Callable, types, args, kwargs, process_group):
    """
    Calls the custom op for this ShardingSpec if it exists.
    """
    class_name = type(sharding_spec).__qualname__
    if not _has_custom_op(sharding_spec, op):
        raise RuntimeError(f'Custom op: {op} not registered for {class_name}')
    func = _CUSTOM_SHARDING_SPEC_OPS[class_name][op]
    return func(types, args, kwargs, process_group)