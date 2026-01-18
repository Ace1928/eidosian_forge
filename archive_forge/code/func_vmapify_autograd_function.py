import torch
from torch._ops import HigherOrderOperator
from torch._C._functorch import TransformType
from torch._functorch.utils import enable_single_level_autograd_function
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch._functorch.vmap import (
from torch._functorch.apis import vmap
from torch._functorch.vmap import _broadcast_to_and_flatten
from torch.autograd.forward_ad import _set_fwd_grad_enabled
from typing import Any, NamedTuple, Tuple
def vmapify_autograd_function(autograd_function, in_dims, batch_size, randomness):
    init_val = 'not populated'
    out_dims = init_val
    input_shapes: Any = init_val
    saved_tensors_bdims: Any = init_val

    def forward(*operands):
        nonlocal out_dims
        outputs, out_dims = restore_vmap(autograd_function.forward, in_dims, batch_size, randomness)(*operands)
        return outputs

    def setup_context(ctx, inputs, outputs):
        input_shapes_ = None
        saved_tensors_bdims_ = None

        def inner(inputs, outputs):
            wrapped_ctx = CtxCustomSave(ctx, current_level())
            autograd_function.setup_context(wrapped_ctx, inputs, outputs)
            nonlocal input_shapes_
            input_shapes_ = tuple((inp.shape if isinstance(inp, torch.Tensor) else None for inp in inputs))
            nonlocal saved_tensors_bdims_
            saved_tensors_bdims_ = wrapped_ctx._pt_saved_tensors_bdims
        restore_vmap(inner, (in_dims, out_dims), batch_size, randomness)(inputs, outputs)
        nonlocal input_shapes
        input_shapes = input_shapes_
        nonlocal saved_tensors_bdims
        saved_tensors_bdims = saved_tensors_bdims_

    def jvp(ctx, *tangents):
        assert out_dims != init_val
        assert saved_tensors_bdims != init_val

        def jvp_no_context(saved_tensors, tangents):
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.jvp(wrapped_ctx, *tangents)
        tangent_in_dims = get_tangents_in_dims(in_dims, tangents)
        out_tangents, out_tangents_dims = restore_vmap(jvp_no_context, (saved_tensors_bdims, tangent_in_dims), batch_size, randomness)(ctx.saved_tensors, tangents)
        result = reductify(out_tangents, out_tangents_dims, out_dims, batch_size)
        return result

    def backward(ctx, *grad_outputs):
        assert out_dims != init_val
        assert input_shapes != init_val
        assert saved_tensors_bdims != init_val

        def backward_no_context(inputs):
            saved_tensors, grad_outputs = inputs
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.backward(wrapped_ctx, *grad_outputs)
        grad_ins, grad_ins_dims = restore_vmap(backward_no_context, ((saved_tensors_bdims, out_dims),), batch_size, randomness)((ctx.saved_tensors, grad_outputs))
        result = reductify(grad_ins, grad_ins_dims, in_dims, batch_size, input_shapes)
        return result
    name = f'Vmapped{autograd_function.__name__}'
    Generated = type(name, (torch.autograd.Function,), {'forward': staticmethod(forward), 'backward': staticmethod(backward), 'jvp': staticmethod(jvp), 'setup_context': staticmethod(setup_context), 'generate_vmap_rule': True})

    def get_out_dims():
        assert out_dims != init_val
        return out_dims
    return (Generated, get_out_dims)