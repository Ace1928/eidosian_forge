import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
@staticmethod
def _extract_leaf_events(op_tree: OpTree) -> Tuple[_ProfilerEvent, ...]:
    """Partially traverse the op tree and extract top level ops.

        Consider the following code:
        ```
        with record_function("My annotation"):
            x.zero_()
            y.zero_()
        ```

        The op tree (assuming no Autograd) will look like:
          <Python context>
            TorchOp: "My annotation"
              TorchOp: zero_
                TorchOp: fill_
              TorchOp: zero_
                TorchOp: fill_

        The recursive structure of operator calls makes data flow unwieldy.
        In order to simplify analysis we would like to select the highest level
        ops to represent in the graph. In this case those are the `zero_` ops;
        the fact that `fill_` is called is an implementation detail. We also
        do not want to group everything under "My annotation" as this could
        create overly coarse bundles and lose critical semantics.

        To address this issue we walk over the graph and select the topmost
        torch ops ** which match at least one operator schema **. These form
        the leaves of the first pass through the op tree. (As well as any
        allocations or frees which do are not part of a kernel.) These events
        form the logical nodes in our data flow graph.
        """
    leaf_events: List[_ProfilerEvent] = []

    def leaf_op(e: _ProfilerEvent) -> bool:
        return e.typed[0] == _EventType.TorchOp and (e.typed[1].scope == RecordScope.BACKWARD_FUNCTION or bool(SchemaMatcher.match_schemas(e.typed[1])))

    def children_fn(e: _ProfilerEvent):
        if leaf_op(e) or e.tag == _EventType.Allocation:
            leaf_events.append(e)
            return []
        return e.children
    for _ in op_tree.dfs(children_fn=children_fn):
        pass
    return tuple(sorted(leaf_events, key=lambda x: x.start_time_ns))