from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple
import torch
from torch import nn
class ShardSyncLayer(torch.autograd.Function):
    """
    The shard sync layer is a synchronization point between model shards.
    - In the forward pass, it drops parameters in the previous shard and
    loads parameters for the next shard.
    - In the backward pass, it does the reverse.
    It does not change or create any outputs at all, instead it just
    forwards the input as the output.
    NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    @_conditional_amp_fwd_decorator
    def forward(ctx: Any, inputs: Any, index: int, model_slices: Any, model_instance: Any) -> Any:
        drop_index = index
        load_index = index + 1
        max_slices = len(model_slices)
        if drop_index >= 0:
            model_slices[drop_index].forward_drop()
        if load_index < max_slices:
            model_slices[load_index].forward_load()
        ctx.index = index
        ctx.model_slices = model_slices
        ctx.model_instance = model_instance
        return inputs if isinstance(inputs, tuple) else (inputs,)

    @staticmethod
    @_conditional_amp_bwd_decorator
    def backward(ctx, *grad_outputs):
        load_index = ctx.index
        drop_index = load_index + 1
        model_slices = ctx.model_slices
        model_instance = ctx.model_instance
        if drop_index == len(model_slices):
            model_instance._activations[-1] = tuple([a.cuda() for a in list(model_instance._activations[-1])])
        if drop_index < len(model_slices):
            model_slices[drop_index].backward_drop()
            model_instance._activations[drop_index] = tuple([a.cpu() for a in list(model_instance._activations[drop_index])])
        if load_index >= 0:
            model_slices[load_index].backward_load()
            model_instance._activations[load_index] = tuple([a.cuda() for a in list(model_instance._activations[load_index])])
        if isinstance(grad_outputs, tuple):
            return (grad_outputs[0], None, None, None)
        return (grad_outputs, None, None, None)