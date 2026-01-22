import dataclasses
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import py_sym_types
class PytreeThunk:
    spec: Optional[pytree.TreeSpec] = None
    is_simple = None
    is_really_simple = None

    def set(self, spec):
        assert self.spec is None or self.spec == spec
        self.spec = spec
        if type(self.spec) in [tuple, list] and all((isinstance(i, pytree.LeafSpec) for i in spec.children_specs)):
            self.is_simple = True
        if isinstance(self.spec, pytree.LeafSpec):
            self.is_really_simple = True

    def unflatten(self, x):
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        assert self.spec is not None
        return pytree.tree_unflatten(x, self.spec)