from typing import Dict
import torch
def _replace_sym_size_ops_pass(gm: torch.fx.GraphModule):
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.target in replacements:
                node.target = replacements[node.target]