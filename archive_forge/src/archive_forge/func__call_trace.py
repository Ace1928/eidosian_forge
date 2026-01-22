import inspect
import operator
from typing import Dict, List, Optional, Tuple, cast
from torch.distributed.nn import RemoteModule
import torch.fx
import torch.nn as nn
from . import PipelineModulesGraph
def _call_trace(tracer: RemoteModuleTracer, module: nn.Module) -> torch.fx.Graph:
    try:
        org_named_modules = RemoteModule.named_modules
        org_named_children = RemoteModule.named_children
        RemoteModule.named_modules = nn.Module.named_modules
        RemoteModule.named_children = nn.Module.named_children
        return tracer.trace(module)
    finally:
        RemoteModule.named_modules = org_named_modules
        RemoteModule.named_children = org_named_children