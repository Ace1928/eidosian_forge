import ast
import inspect
import textwrap
import copy
import functools
from types import FunctionType
from typing import cast, Union, Callable, Dict, Optional, Any
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph import Graph
from torch._sources import normalize_source_lines
import torch
class RewritingTracer(Tracer):

    def trace(self, root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]]=None) -> Graph:
        return super().trace(_rewrite(root), concrete_args)