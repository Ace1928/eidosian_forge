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
class _Patcher:

    def __init__(self):
        super().__init__()
        self.patches_made: List[_PatchedFn] = []
        self.visited: Set[int] = set()

    def patch(self, frame_dict: Dict[str, Any], name: str, new_fn: Callable, deduplicate: bool=True):
        """
        Replace frame_dict[name] with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None))
        elif getattr(frame_dict[name], '__fx_already_patched', False):
            return
        else:
            self.patches_made.append(_PatchedFnSetItem(frame_dict, name, frame_dict[name]))
        frame_dict[name] = new_fn

    def patch_method(self, cls: type, name: str, new_fn: Callable, deduplicate: bool=True):
        """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, '__fx_already_patched', False):
            return
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
        setattr(cls, name, new_fn)

    def visit_once(self, thing: Any):
        """Return True on the first call to with thing, otherwise false"""
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Undo all the changes made via self.patch() and self.patch_method()
        """
        while self.patches_made:
            self.patches_made.pop().revert()
        self.visited.clear()