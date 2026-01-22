import dataclasses
import functools
from importlib import import_module
from typing import Any, List, Optional
from functorch.compile import min_cut_rematerialization_partition
import torch
from torch import _guards
from torch._functorch.compilers import ts_compile
from .common import aot_autograd
from .registry import register_debug_backend as register_backend
class ExplainWithBackend:
    """
    This class is intended to be used as a backend for `torch.compile`. It is
    composable with other backends. When used in this way, it accumulates
    information about graph breaks, ops, and other info and provides a string
    representation summarizing this information.

    Attributes:
        backend (str): The name of the backend to use for optimization.
        graphs (list): A list of the graphs captured by TorchDynamo.
        op_count (int): The total number of operations in all optimized graphs.
        break_reasons (list): A list of graph break reasons with stack traces.

    Example Usage:
        def fn(x):
            x = torch.sigmoid(x)
            return x

        torch._dynamo.reset()
        eb = ExplainWithBackend("inductor")
        optimized_fn = torch.compile(fn, backend=eb)
        result = optimized_fn(torch.randn(5))
        print(eb.output())
    """

    def __init__(self, backend):
        from .registry import lookup_backend
        self.backend = lookup_backend(backend)
        self.graphs = []
        self.op_count = 0
        self.break_reasons = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        gm, self.graphs, self.op_count, _, self.break_reasons = _explain_graph_detail(gm, self.graphs, self.op_count, [], self.break_reasons)
        return self.backend(gm, example_inputs)

    def output(self) -> ExplainOutput:
        graph_count = len(self.graphs)
        output = ExplainOutput(self.graphs, graph_count, graph_count - 1, self.break_reasons, self.op_count)
        return output