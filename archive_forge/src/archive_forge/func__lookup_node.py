import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def _lookup_node(self, node: fx.Node, graph: fx.Graph) -> Optional[fx.Node]:
    if graph == self.setup_graph:
        return self._setup_mapping.get(node, None)
    elif graph == self.cleanup_graph:
        return self._cleanup_mapping.get(node, None)
    return node