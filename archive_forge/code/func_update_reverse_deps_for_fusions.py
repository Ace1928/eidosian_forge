import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def update_reverse_deps_for_fusions(self, deps: Dict[torch.fx.Node, NodeSet]):
    processed_node = set()
    for node, fusion in self.fusions.items():
        if node in processed_node:
            continue
        new_dep = set()
        for n in fusion:
            new_dep.update(deps[n])
        new_dep.difference_update(fusion)
        for n in fusion:
            deps[n] = new_dep
            for arg in n.all_input_nodes:
                if arg not in fusion:
                    deps[arg].update(fusion)
            processed_node.add(n)