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
def _insert_prim_assert_inplace(self, graph, node: torch.fx.Node, value: Any):
    assert_msg = f'Input {node.name} is specialized to be {value} at tracing time,it is not supported to pass in a different value at run time.'
    with graph.inserting_after(node):
        eq_node = graph.call_function(operator.eq, (node, value))
    with graph.inserting_after(eq_node):
        tensor_eq_node = graph.call_function(torch.ops.aten.scalar_tensor.default, (eq_node,))
    with graph.inserting_after(tensor_eq_node):
        _ = graph.call_function(torch.ops.aten._assert_async.msg, (tensor_eq_node, assert_msg))