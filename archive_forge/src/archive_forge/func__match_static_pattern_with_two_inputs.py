import torch
from torch.fx import map_arg, Node
from torch.fx.graph import Graph
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.fx import GraphModule
from .utils import (
from ..utils import _parent_name
from ..qconfig import QConfigAny
from ..quantization_mappings import get_quantized_operator
from .utils import create_node_from_old_node_preserve_meta
from typing import Dict, Tuple, Type, List, Callable, Any, Union, Set, Optional
import operator
def _match_static_pattern_with_two_inputs(node: Node, modules: Dict[str, nn.Module], qconfig_map: Dict[str, QConfigAny], matching_modules_or_ops: List[Callable]) -> Union[Tuple[Node, Node], Tuple[None, None]]:
    """
                      (dequantize     Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 2-tuple of:
      1) q_node: the quantize node,
      2) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 2-tuple of (None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
    """
    SKIP_LOWERING_VALUE = (None, None)
    if node.op != 'call_function' or node.target != torch.quantize_per_tensor:
        return SKIP_LOWERING_VALUE
    q_node = node
    ref_node = q_node.args[0]
    assert isinstance(ref_node, Node)
    if should_skip_lowering(ref_node, qconfig_map):
        return SKIP_LOWERING_VALUE
    if isinstance(matching_modules_or_ops[0], type) and issubclass(matching_modules_or_ops[0], nn.Module):
        expected_op = 'call_module'
        match_key = type(_get_module(ref_node, modules))
    else:
        return SKIP_LOWERING_VALUE
    if ref_node.op != expected_op or match_key not in matching_modules_or_ops:
        return SKIP_LOWERING_VALUE
    if len(ref_node.args) != 2:
        return SKIP_LOWERING_VALUE
    for i in range(len(ref_node.args)):
        arg = ref_node.args[i]
        if not is_dequantize_node(arg):
            return SKIP_LOWERING_VALUE
    return (q_node, ref_node)