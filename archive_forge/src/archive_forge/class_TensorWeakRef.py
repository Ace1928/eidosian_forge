from __future__ import annotations
import weakref
from weakref import ref
from _weakrefset import _IterationGuard  # type: ignore[attr-defined]
from collections.abc import MutableMapping, Mapping
from torch import Tensor
import collections.abc as _collections_abc
class TensorWeakRef:
    """Wrapper around a weak ref of a Tensor that handles the _fix_weakref() call required when unwrapping a Tensor weakref."""
    ref: WeakRef[Tensor]

    def __init__(self, tensor: Tensor):
        assert isinstance(tensor, Tensor)
        self.ref = weakref.ref(tensor)

    def __call__(self):
        out = self.ref()
        if out is None:
            return out
        assert isinstance(out, Tensor)
        out._fix_weakref()
        return out