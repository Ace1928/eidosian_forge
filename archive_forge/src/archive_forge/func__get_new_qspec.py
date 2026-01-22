import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import (
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import (
from .utils import (
def _get_new_qspec(qspec: QuantizationSpecBase):
    if isinstance(qspec, SharedQuantizationSpec):
        new_edge_or_node = _get_new_edge_or_node(qspec.edge_or_node)
        return SharedQuantizationSpec(new_edge_or_node)
    elif isinstance(qspec, DerivedQuantizationSpec):
        new_derived_from = [_get_new_edge_or_node(x) for x in qspec.derived_from]
        return dataclasses.replace(qspec, derived_from=new_derived_from)
    else:
        return qspec