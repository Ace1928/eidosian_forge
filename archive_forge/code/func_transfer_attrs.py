import builtins
import copy
import functools
import inspect
import math
import os
import warnings
import collections
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import (
import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from ._compatibility import compatibility
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, TracerBase, Scope, ScopeContextManager
def transfer_attrs(fr, to):
    for attr_name in dir(fr):
        attr_val = getattr(fr, attr_name)
        if not callable(attr_val) and (not attr_name.startswith('__')) and (not hasattr(to, attr_name)):
            setattr(to, attr_name, attr_val)