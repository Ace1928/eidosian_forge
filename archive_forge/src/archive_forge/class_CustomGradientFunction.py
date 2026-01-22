import contextlib
import os
import ml_dtypes
import numpy as np
import torch
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.config import floatx
from keras.src.utils import tree
class CustomGradientFunction(torch.autograd.Function):
    """Enables custom forward & backward passes for gradient computation."""

    @staticmethod
    def forward(ctx, forward_fn, *args, **kwargs):
        """Forward pass computation specification.

        Args:
            ctx: Context object.
            forward_fn: Function to compute forward pass.
            *args: Arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.
        """
        ctx.forward_fn = forward_fn
        ctx.save_for_backward(*args)
        try:
            output, ctx.grad_fn = forward_fn(*args, **kwargs)
        except:
            output = forward_fn(*args, **kwargs)
            ctx.grad_fn = lambda *args, **kwargs: torch.full((), float('nan'))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computation specification.

        Args:
            ctx: Context object.
            grad_output: Gradient with respect to the output.
        """
        args = ctx.saved_tensors
        grad_fn = ctx.grad_fn
        if grad_fn is None:
            raise ValueError('grad_fn must be provided for custom gradient')
        grads = grad_fn(*args, upstream=grad_output)
        if not isinstance(grads, tuple):
            grads = (grads,)
        return (None,) + grads