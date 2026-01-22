import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
from torch.fx import (
from torch.fx.node import Argument
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
def _maybe_insert_input_and_output_observers_for_node(node: Node, model: torch.fx.GraphModule, obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize], is_qat: bool):
    this_node_quantization_annotation = node.meta['quantization_annotation'] if 'quantization_annotation' in node.meta else None
    if this_node_quantization_annotation is None:
        return
    named_modules = dict(model.named_modules(remove_duplicate=False))
    _maybe_insert_input_observers_for_node(node, None, model, named_modules, obs_or_fq_map, is_qat)
    output_is_a_tensor = 'val' in node.meta and isinstance(node.meta['val'], FakeTensor)
    if not output_is_a_tensor:
        return
    maybe_output_obs_node = _maybe_insert_output_observer_for_node(node, model, named_modules, model.graph, obs_or_fq_map, is_qat)
    if maybe_output_obs_node is None:
        return
    orig_users = list(node.users.keys())
    for user_node in orig_users:
        if user_node is maybe_output_obs_node:
            continue
        user_node.replace_input_with(node, maybe_output_obs_node)