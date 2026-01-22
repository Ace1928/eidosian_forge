from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple
import torch
from torch import nn
class OffloadFunction(torch.autograd.Function):
    """
    This Function enables checkpointing of intermediate activations at
    shard boundaries by overriding the forward and backward pass of the nn.Module.

    - In the FW pass, it drops parameters in the previous shard and
    loads parameters for the next shard. No graph is constructed in the FW pass.
    This enables us to offload intermediate activations present at the shard
    boundaries.

    - In the BW pass, it does the reverse. We run the forward pass using the
    saved intermediate activations and calculate gradients as needed.
    The trade-off is latency vs memory when using activation checkpointing.

    - Follows heavily from https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html#checkpoint.

    NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    @_conditional_amp_fwd_decorator
    def forward(ctx: Any, inputs: Any, dummy_input: Any, model_instance: Any) -> Any:
        inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        ctx.inputs = inputs
        ctx.model_instance = model_instance
        ctx.grad_requirements = tuple((x.requires_grad for x in inputs))
        ctx.fwd_rng_state = torch.get_rng_state()
        model_instance._activations = [inputs]
        for index, layer_shard in enumerate(model_instance.model_slices):
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:forward_load'):
                model_instance._activations[index] = tuple([a.cuda() for a in list(model_instance._activations[index])])
                layer_shard.forward_load()
            inputs = model_instance._activations[index]
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:no_grad_forward_pass'):
                with torch.no_grad():
                    output_list: List[Any] = []
                    for given_input in inputs:
                        given_input_list = torch.chunk(given_input, model_instance._num_microbatches)
                        given_output_list = []
                        for inputs in given_input_list:
                            output = layer_shard(inputs)
                            given_output_list.append(output)
                        given_output = torch.cat(given_output_list).squeeze(-1)
                        output_list.append(given_output)
                    output = tuple(output_list)
            output = output if isinstance(output, tuple) else (output,)
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:forward_drop'):
                model_instance._activations[index] = tuple([a.cpu() for a in list(model_instance._activations[index])])
                model_instance._activations.append(output)
                layer_shard.forward_drop()
        result = model_instance._activations[-1]
        result = [r.cuda() for r in result]
        for r in result:
            r.requires_grad = True
        return result[0] if len(result) == 1 else result

    @staticmethod
    @_conditional_amp_bwd_decorator
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('Checkpointing is not compatible with .grad(), please use .backward() if possible')
        inputs = ctx.inputs
        model_instance = ctx.model_instance
        for i, need_grad in enumerate(ctx.grad_requirements):
            inputs[i].requires_grad = need_grad
        all_grads = [grad_outputs]
        for model_shard, activation in zip(reversed(model_instance.model_slices), reversed(model_instance._activations[:-1])):
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:backward_load'):
                activation = tuple([a.cuda() for a in list(activation)])
                model_shard.backward_load()
            bwd_rng_state = torch.get_rng_state()
            activation = torch.utils.checkpoint.detach_variable(activation)
            final_grads = all_grads[-1]
            if isinstance(activation, torch.Tensor):
                activation = (activation,)
            if isinstance(final_grads, torch.Tensor):
                final_grads = (final_grads,)
            chunked_grad_list: List[Any] = []
            for chunked_activation, chunked_grad in zip(torch.chunk(*activation, model_instance._num_microbatches), torch.chunk(*final_grads, model_instance._num_microbatches)):
                torch.set_rng_state(ctx.fwd_rng_state)
                if isinstance(chunked_activation, torch.Tensor):
                    chunked_activation = (chunked_activation,)
                if isinstance(chunked_grad, torch.Tensor):
                    chunked_grad = (chunked_grad,)
                for a in chunked_activation:
                    if a.dtype == torch.long:
                        continue
                    a.requires_grad = True
                    a.retain_grad()
                with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:forward_pass_with_enable_grad'):
                    with torch.enable_grad():
                        outputs = model_shard(*chunked_activation)
                torch.set_rng_state(bwd_rng_state)
                with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:backward_pass'):
                    torch.autograd.backward(outputs, chunked_grad)
                intermediate_grads = []
                for a in chunked_activation:
                    if a.grad is not None:
                        intermediate_grads.append(a.grad)
                if None not in intermediate_grads:
                    chunked_grad_list += intermediate_grads
            if chunked_grad_list:
                all_grads.append(torch.cat(chunked_grad_list).squeeze(-1))
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:backward_drop'):
                model_shard.backward_drop()
        detached_inputs = model_instance._activations[0]
        grads = tuple((inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs))
        return (None, None) + grads