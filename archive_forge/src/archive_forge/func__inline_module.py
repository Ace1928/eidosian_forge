import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module
def _inline_module(gm: torch.fx.GraphModule, inline_mod_name: str):
    """
    Given `gm` and some graph module which is called with target name `inline_mod_name`,
    this helper will inline all of the nodes from that called graph module into `gm`.
    """
    inline_mod = dict(gm.named_modules())[inline_mod_name]
    assert isinstance(inline_mod, torch.fx.GraphModule)
    call_mod_node_to_replace = None
    for node in gm.graph.nodes:
        if node.op == 'call_module' and node.target == inline_mod_name:
            call_mod_node_to_replace = node
            break
    assert call_mod_node_to_replace is not None
    call_mod_args = call_mod_node_to_replace.args
    replacement_mapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    ph_count = 0

    def replacement_fn(node):
        new_node = replacement_mapping[node]
        new_node.meta = node.meta.copy()
        return new_node
    for inline_node in inline_mod.graph.nodes:
        if inline_node.op == 'placeholder':
            replacement_mapping[inline_node] = call_mod_args[ph_count]
            ph_count += 1
            continue
        if inline_node.op == 'output':
            outputs = inline_node.args[0]
            output_replacements = map_arg(outputs, replacement_fn)
            call_mod_node_to_replace.replace_all_uses_with(output_replacements)
            continue
        with gm.graph.inserting_before(call_mod_node_to_replace):
            new_node = gm.graph.node_copy(inline_node, replacement_fn)
        replacement_mapping[inline_node] = new_node
    gm.graph.eliminate_dead_code()