from .graph_module import GraphModule
from .graph import Graph
from .node import Argument, Node, Target, map_arg, map_aggregate
from .proxy import Proxy
from ._symbolic_trace import Tracer
from ._compatibility import compatibility
from . import config
import torch.fx.traceback as fx_traceback
import torch
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import inspect
from contextlib import contextmanager
from torch.hub import tqdm
@contextmanager
def _set_current_node(self, node):
    with fx_traceback.set_current_meta(node):
        yield