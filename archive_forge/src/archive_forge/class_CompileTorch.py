import functools
from .autoray import (
from . import lazy
class CompileTorch:
    """ """

    def __init__(self, fn, **kwargs):
        import torch
        self.torch = torch
        if not hasattr(fn, '__name__') and isinstance(fn, functools.partial):
            functools.update_wrapper(fn, fn.func)
        self._fn = fn
        self._jit_fn = None
        kwargs.setdefault('check_trace', False)
        self._jit_kwargs = kwargs

    def setup(self, *args, **kwargs):
        flat_tensors, ref_tree = tree_flatten((args, kwargs), get_ref=True)

        def flat_fn(flat_tensors):
            args, kwargs = tree_unflatten(flat_tensors, ref_tree)
            return self._fn(*args, **kwargs)
        self._jit_fn = self.torch.jit.trace(flat_fn, [flat_tensors], **self._jit_kwargs)

    def __call__(self, *args, array_backend=None, **kwargs):
        if array_backend != 'torch':
            args = tree_map(self.torch.as_tensor, args, is_array)
        if self._jit_fn is None:
            self.setup(*args, **kwargs)
        out = self._jit_fn(tree_flatten((args, kwargs)))
        if array_backend != 'torch':
            out = do('asarray', out, like=array_backend)
        return out