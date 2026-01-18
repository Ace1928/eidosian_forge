import contextlib
import functools
from typing import List, Optional
import torch
from torch._dynamo.external_utils import call_hook
from torch._dynamo.source import GetItemSource, LocalSource
from torch._dynamo.utils import counters, lazy_format_graph_code
from torch._logging import getArtifactLogger
from torch._prims_common import clone_preserve_strides
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import (
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.fx.proxy import Proxy
def to_proxy(self, t):
    if t is None:
        return None
    if isinstance(t, list):
        return [self.to_proxy(x) for x in t]
    if isinstance(t, tuple):
        return tuple((self.to_proxy(x) for x in t))
    assert isinstance(t, (torch.Tensor, torch.SymInt))
    return fetch_tensor_proxy(self.fx_tracer)(t).proxy