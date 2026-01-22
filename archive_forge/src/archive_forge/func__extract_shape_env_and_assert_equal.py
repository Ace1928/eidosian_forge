import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def _extract_shape_env_and_assert_equal(args, kwargs):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymTypes

    def assert_equal(old: Optional[ShapeEnv], new: ShapeEnv) -> ShapeEnv:
        if old is not None:
            assert old is new, 'call with different ShapeEnv'
        return new
    shape_env = None
    for val in itertools.chain(args, kwargs.values()):
        if isinstance(val, ShapeEnv):
            shape_env = assert_equal(shape_env, val)
        if isinstance(val, SymTypes):
            shape_env = assert_equal(shape_env, val.node.shape_env)
    return shape_env