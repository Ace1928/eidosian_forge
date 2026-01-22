from functools import partial
from typing import Optional
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup
from flash_attn.ops.activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd
from flash_attn.utils.distributed import (
class FusedMLPFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight1, bias1, weight2, bias2, activation='gelu_approx', save_pre_act=True, return_residual=False, checkpoint_lvl=0, heuristic=0, process_group=None, sequence_parallel=True):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather of x before doing the matmul.
        If sequence_parallel=False, then the input is already gathered.

        checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out / relu_out in the bwd
        2: recompute pre_act and gelu_out / relu_out in the bwd
        """
        assert -1 <= heuristic <= 4
        assert activation in ['gelu_approx', 'relu', 'sqrelu']
        if activation == 'sqrelu':
            assert heuristic == -1
        if not save_pre_act:
            checkpoint_lvl = 2
        assert checkpoint_lvl in [0, 1, 2]
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.activation = activation
        ctx.heuristic = heuristic
        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            weight1, weight2 = [a.to(dtype=dtype) for a in [weight1, weight2]]
            bias1 = bias1.to(dtype=dtype) if bias1 is not None else None
            bias2 = bias2.to(dtype=dtype) if bias2 is not None else None
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous() if bias1 is not None else None
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous() if bias2 is not None else None
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = (total_x.shape[:-1], total_x.shape[-1])
        batch_dim = batch_shape.numel()
        if min(batch_dim, n, *weight1.shape, *weight2.shape) > 65535 * 32:
            raise RuntimeError('fused_dense only supports matrix dims <= 2M')
        if heuristic == -1:
            pre_act = F.linear(total_x, weight1, bias1)
            activation_fn = partial(F.gelu, approximate='tanh') if activation == 'gelu_approx' else sqrelu_fwd if activation == 'sqrelu' else F.relu
            with torch.jit.fuser('fuser2'):
                output1 = activation_fn(pre_act)
        else:
            is_gelu = activation == 'gelu_approx'
            output1, *rest = fused_dense_cuda.linear_act_forward(total_x.reshape(batch_dim, n), weight1, bias1, is_gelu, save_pre_act, heuristic)
            if save_pre_act:
                pre_act = rest[0]
        output2 = F.linear(output1, weight2, bias2)
        if checkpoint_lvl == 0 or (checkpoint_lvl == 1 and activation == 'relu'):
            ctx.save_for_backward(x, weight1, weight2, pre_act, output1)
        elif checkpoint_lvl == 1:
            ctx.save_for_backward(x, weight1, weight2, pre_act)
        elif checkpoint_lvl == 2:
            ctx.save_for_backward(x, weight1, weight2, bias1)
        output2 = output2.reshape(*batch_shape, output2.shape[-1])
        return output2 if not return_residual else (output2, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        activation = ctx.activation
        activation_fn = partial(F.gelu, approximate='tanh') if activation == 'gelu_approx' else sqrelu_fwd if activation == 'sqrelu' else F.relu
        if ctx.return_residual:
            grad_input, = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        x, weight1, weight2, *rest = ctx.saved_tensors
        if process_group is None or not sequence_parallel:
            total_x = x
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        if checkpoint_lvl in [0, 1]:
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            if checkpoint_lvl == 0 or (checkpoint_lvl == 1 and activation == 'relu'):
                pre_act, output1 = rest
            elif checkpoint_lvl == 1:
                pre_act, = rest
                with torch.jit.fuser('fuser2'):
                    output1 = activation_fn(pre_act)
        elif checkpoint_lvl == 2:
            bias1, = rest
            if process_group is not None and sequence_parallel:
                total_x, _ = all_gather_raw(x, process_group)
            if ctx.heuristic == -1:
                pre_act = F.linear(total_x, weight1, bias1)
                with torch.jit.fuser('fuser2'):
                    output1 = activation_fn(pre_act)
            else:
                output1, pre_act = fused_dense_cuda.linear_act_forward(total_x.reshape(batch_dim, total_x.shape[-1]), weight1, bias1, activation == 'gelu_approx', True, ctx.heuristic)
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        output1 = output1.reshape(batch_dim, output1.shape[-1])
        pre_act = pre_act.reshape(batch_dim, pre_act.shape[-1])
        if ctx.needs_input_grad[3]:
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output, ctx.needs_input_grad[4])
        else:
            grad_weight2 = None
            grad_bias2 = grad_output if ctx.needs_input_grad[4] else None
        if ctx.heuristic == -1:
            grad_output1 = F.linear(grad_output, weight2.t())
            activation_grad_fn = gelu_bwd if activation == 'gelu_approx' else sqrelu_bwd if activation == 'sqrelu' else relu_bwd
            with torch.jit.fuser('fuser2'):
                grad_pre_act = activation_grad_fn(grad_output1, pre_act)
        else:
            grad_pre_act, grad_bias1 = fused_dense_cuda.bias_act_linear_dgrad_bgrad(weight2, grad_output, pre_act, activation == 'gelu_approx', ctx.heuristic)
            if not ctx.needs_input_grad[2]:
                grad_bias1 = None
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_pre_act, weight1.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]), grad_pre_act, weight1)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
                grad_input, handle_grad_input = reduce_fn(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.heuristic == -1:
            if ctx.needs_input_grad[1]:
                if process_group is not None and sequence_parallel and (checkpoint_lvl != 2):
                    handle_x.wait()
                grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_wgrad(total_x.reshape(batch_dim, total_x.shape[-1]), grad_pre_act, ctx.needs_input_grad[2])
            else:
                grad_weight1 = None
                grad_bias1 = grad_pre_act if ctx.needs_input_grad[2] else None
        elif ctx.needs_input_grad[1]:
            if process_group is not None and sequence_parallel and (checkpoint_lvl != 2):
                handle_x.wait()
            grad_weight1 = F.linear(grad_pre_act.t(), total_x.reshape(batch_dim, total_x.shape[-1]).t())
        else:
            grad_weight1 = None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return (grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2, None, None, None, None, None, None, None)