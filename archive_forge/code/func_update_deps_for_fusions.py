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
def update_deps_for_fusions(self):
    """
        Updates graph of dependencies so that:
        - nodes from the same fusion depend on the same set of outer nodes,
        - outer nodes depending on a fusion depend on all nodes in that fusion.
        """
    for node in self.fusions:
        fusion = self.fusions[node]
        for fused_neighbor in fusion:
            self.deps[node].update(self.deps[fused_neighbor] - fusion)
            for user in fused_neighbor.users:
                if user not in fusion:
                    self.deps[user].add(node)