import collections
import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.ns.fx.mappings import (
from torch.ao.ns.fx.graph_matcher import (
from .fx.weight_utils import (
from .fx.graph_passes import (
from .fx.utils import (
from .fx.ns_types import (
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_root_node_getter
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.fx.match_utils import _find_matches
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from torch.ao.quantization.fx.qconfig_mapping_utils import _generate_node_name_to_qconfig
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization import QConfigMapping
from torch.ao.ns.fx.n_shadows_utils import (
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from typing import Dict, Tuple, Callable, List, Optional, Set, Any, Type
def _prepare_n_shadows_add_loggers_model(model: torch.nn.Module, example_inputs: Any, qconfig_mapping: QConfigMapping, backend_config: BackendConfig) -> torch.nn.Module:
    """
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.

    This creates a model which provides logging for the following
    problem: if we quantize `model` with `qconfig_mapping` and feed
    the same input through both models, log the comparisons of
    corresponding intermediate layers.

    The problem is solved with a single model.  Specifically, we
    partition `model` into N subgraphs, create a copy of each relevant
    subgraph, wrap it in a module, apply the quantization API to that
    module, and hook up loggers to measure the comparisons.

    Example starting graph:

      x0 -> op0 -> x1 -> op1 -> x2

    Example config: quantize op0 to int8, do nothing to op1.
    The following graph will be created:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \\                        \\                           \\       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_0 -> x2_1 ----> clog

    Where op0_0 is op0, op0_1 is op0 wrapped in a submodule and quantized
    to int8, op1_0 is op1 (appearing in the graph twice), log is a logger,
    and clog is a comparison logger.
    """
    tracer = quantize_fx.QuantizationTracer([], [])
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    mt._node_name_to_scope = tracer.node_name_to_scope
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: List[str] = []
    standalone_module_classes: List[Type] = []
    custom_module_classes: List[Type] = []
    matches = _find_matches(mt.graph, modules, patterns, root_node_getter_mapping, standalone_module_names, standalone_module_classes, custom_module_classes)
    subgraphs_dedup: Dict[str, List[Node]] = _get_dedup_subgraphs(matches)
    node_name_to_qconfig = _generate_node_name_to_qconfig(mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope)
    create_add_loggers_graph(mt, subgraphs_dedup, qconfig_mapping, node_name_to_qconfig)
    return mt