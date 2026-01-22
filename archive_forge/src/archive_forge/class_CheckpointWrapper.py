import warnings
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
class CheckpointWrapper(ActivationWrapper):
    """
    An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """

    def __init__(self, mod: torch.nn.Module, checkpoint_impl: CheckpointImpl=CheckpointImpl.NO_REENTRANT, checkpoint_fn=None, **checkpoint_fn_kwargs):
        super().__init__(mod)
        self.checkpoint_impl = checkpoint_impl
        if checkpoint_fn is None:
            self.checkpoint_fn = partial(torch_utils_checkpoint, use_reentrant=self.checkpoint_impl == CheckpointImpl.REENTRANT, **checkpoint_fn_kwargs)
        else:
            self.checkpoint_fn = partial(checkpoint_fn, **checkpoint_fn_kwargs)

    def forward(self, *args, **kwargs):
        if self.checkpoint_impl == CheckpointImpl.REENTRANT and kwargs != {}:
            flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)

            def my_function(*inputs):
                unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                return self._checkpoint_wrapped_module(*unpacked_args, **unpacked_kwargs)
            return self.checkpoint_fn(my_function, *flat_args)
        else:
            return self.checkpoint_fn(self._checkpoint_wrapped_module, *args, **kwargs)