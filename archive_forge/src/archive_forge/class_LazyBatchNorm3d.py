from typing import Optional, Any
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer
from .. import functional as F
from .. import init
from ._functions import SyncBatchNorm as sync_batch_norm
from .lazy import LazyModuleMixin
from .module import Module
class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
    """A :class:`torch.nn.BatchNorm3d` module with lazy initialization of
    the ``num_features`` argument of the :class:`BatchNorm3d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """
    cls_to_become = BatchNorm3d

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f'expected 5D input (got {input.dim()}D input)')