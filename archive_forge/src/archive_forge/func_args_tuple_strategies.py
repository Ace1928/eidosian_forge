from typing import List, Tuple
import torch
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def args_tuple_strategies(args_schema: Tuple[object, ...]) -> List[TupleStrategy]:
    first_arg = args_schema[0]
    assert isinstance(first_arg, TupleStrategy)
    strategy_len = len(first_arg.childs)
    tuple_strategies: List[TupleStrategy] = []
    for arg in args_schema:
        if isinstance(arg, TupleStrategy):
            assert len(arg.childs) == strategy_len
            tuple_strategies.append(arg)
        elif isinstance(arg, OpStrategy):
            raise RuntimeError('foreach list op only supports tuple strategy!')
    return tuple_strategies