import functools
from typing import List
import torch
import torch.distributed._shard.sharding_spec as shard_spec
from .api import (
from .metadata import ShardMetadata  # noqa: F401
from torch.distributed._shard.op_registry_utils import _decorator_func
from ._ops import *  # noqa: F403
def _sharded_op_impl(func):
    """
    Decorator to register a default sharded op.
    """
    return functools.partial(_decorator_func, op=func, op_table=_SHARDED_OPS)