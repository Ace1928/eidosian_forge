import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch
from torch._export import ExportedProgram
from torch.utils._pytree import (
def _check_input_constraints_for_graph(graph: torch.fx.Graph, range_constraints, equality_constraints):
    from torch._export.passes.add_runtime_assertions_for_constraints_pass import _AddRuntimeAssertionsForConstraintsPass

    def inner(*args):
        _assertion_graph = torch.fx.GraphModule({}, torch.fx.Graph())
        for p in graph.nodes:
            if p.op != 'placeholder':
                continue
            new_p = _assertion_graph.graph.placeholder(p.name)
            new_p.meta = p.meta
        _assertion_graph.graph.output(())
        _assertion_graph_res = _AddRuntimeAssertionsForConstraintsPass(range_constraints, equality_constraints)(_assertion_graph)
        assert _assertion_graph_res is not None
        _assertion_graph = _assertion_graph_res.graph_module
        _assertion_graph(*args)
    return inner