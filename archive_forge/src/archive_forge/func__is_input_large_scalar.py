import itertools
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization.quantizer.utils import (
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
def _is_input_large_scalar(node: Node, gm: torch.fx.GraphModule):
    """Check if input is a large scalar value. So that we can skip quantization for the node
    since histc op (in HistogramObserver) only works for values up to certain upper bound
    """
    if node.op == 'get_attr':
        tensor = getattr(gm, node.target)
        HISTC_UPPER_BOUND = 3402823500000000.0
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    return False