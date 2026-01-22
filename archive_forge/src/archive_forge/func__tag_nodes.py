import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def _tag_nodes(self, selected_nodes: NodeSet):
    """
        Tag selected nodes with tag "minimize". Nodes with the same tags will
        be split to the same submodule afterwards.

        Args:
            selected_nodes: Nodes that we want to minimize. We will tag those nodes
                with "minimize", all preceding nodes with "main_0" and all following
                nodes with "main_1".
        """
    for node in self.module.graph.nodes:
        if node.op not in CALLABLE_NODE_OPS:
            continue
        if node in selected_nodes:
            node.tag = 'minimize'
        elif any((n.tag in {'minimize', 'main_1'} for n in node.all_input_nodes if n.op in CALLABLE_NODE_OPS)):
            node.tag = 'main_1'
        else:
            node.tag = 'main_0'