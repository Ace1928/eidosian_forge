import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer
from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode
import torch
from torch.nn import *
def _copy_attr(from_module: torch.nn.Module, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split('.')
    for item in prefix:
        f = getattr(from_module, item)
        t = getattr(to_module, item, None)
        if f is t:
            return
        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        from_module, to_module = (f, t)
    orig = getattr(from_module, field)
    if isinstance(orig, torch.Tensor) and (not isinstance(orig, torch.nn.Parameter)):
        to_module.register_buffer(field, orig)
    else:
        setattr(to_module, field, orig)