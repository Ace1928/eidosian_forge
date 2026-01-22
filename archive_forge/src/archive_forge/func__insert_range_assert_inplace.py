import copy
import math
import operator
import traceback
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Set, Tuple
import sympy
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import SymInt
from torch._export.pass_base import _ExportPassBase, ProxyValue, PassResult
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._sympy.value_ranges import ValueRanges
def _insert_range_assert_inplace(self, graph: torch.fx.Graph, input_dim: InputDim, dim_node: torch.fx.Node, range: ValueRanges):
    """
        Add runtime asserts for user-specified range constraints for
        each placeholder's dynamic dimension.
        """
    min_val, max_val = _convert_range_to_int(range)
    assert_msg = f'Input {input_dim.input_name}.shape[{input_dim.dim}] is outside of specified dynamic range [{min_val}, {max_val}]'
    with graph.inserting_after(dim_node):
        if min_val > 2:
            self._insert_assert_async_inplace(graph, operator.ge, (dim_node, min_val), assert_msg)
        if max_val < math.inf:
            self._insert_assert_async_inplace(graph, operator.le, (dim_node, max_val), assert_msg)