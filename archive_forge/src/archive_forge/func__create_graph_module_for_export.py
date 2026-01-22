import copy
from collections import defaultdict
import dataclasses
from typing import Dict, List, Optional, Tuple
import warnings
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import SymInt
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRanges
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
from torch.export.graph_signature import (
from torch.export.exported_program import (
from .utils import _check_input_constraints_pre_hook
def _create_graph_module_for_export(root, graph):
    try:
        gm = torch.fx.GraphModule(root, graph)
    except SyntaxError:
        warnings.warn('Unable to execute the generated python source code from the graph. The graph module will no longer be directly callable, but you can still run the ExportedProgram, and if needed, you can run the graph module eagerly using torch.fx.Interpreter.')
        gm = torch.fx.GraphModule(root, torch.fx.Graph())
        gm._graph = graph
    return gm