from typing import Any, Dict, Optional, Tuple, Union
import warnings
import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY
from .fx.tracer import QuantizationTracer
from .fx.tracer import (  # noqa: F401
from .fx.fuse import fuse  # noqa: F401
from .fx.prepare import prepare  # noqa: F401
from .fx.convert import convert
from .backend_config import (  # noqa: F401
from .fx.graph_module import ObservedGraphModule  # noqa: F401
from .fx.custom_config import (
from .fx.utils import get_custom_module_class_keys  # noqa: F401
from .fx.utils import get_skipped_module_name_and_classes
from .qconfig_mapping import QConfigMapping
def _attach_meta_to_node_if_not_exist(model: GraphModule) -> None:
    """ Attach meta field to all nodes of the graph if it does not exist,
    meta field is a field stores some meta information about the node, such
    as dtype and shape information for output of the node, this only exists
    if the program is captured by make_fx (used in quantize_pt2e flow), if
    the program is captured by torch.fx symbolic tracing, this field may not exist,
    so we add it here to avoid checking this all over the places
    """
    for node in model.graph.nodes:
        if not hasattr(node, 'meta'):
            node.meta = {}