from typing import Optional
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
from flash_attn.ops.triton.k_activations import (
def triton_linear_act(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]=None, activation: str='id', save_act_input: bool=False) -> torch.Tensor:
    """
    Compute e = activation(x @ weight.T + bias).
    This wrapper kicks the `kernel_fwd` Triton kernel
    :param x: input tensor
    :param weight: weight matrix
    :param bias: an optional bias tensor
    :param activation: Activation name. Needs to be a Triton kernel.
    :param act_input: an optional tensor to save the activation inputs (for backward)
    :return: result tensor
    """
    assert activation in ['id', 'gelu', 'gelu_approx', 'squared_relu']
    batch_shape, n = (x.shape[:-1], x.shape[-1])
    batch_dim = batch_shape.numel()
    x_reshaped = x.reshape(batch_dim, n)
    if x_reshaped.stride(0) > 1 and x_reshaped.stride(1) > 1:
        x_reshaped = x_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    assert x.dtype == weight.dtype, f'Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}'
    if bias is not None:
        assert x.dtype == bias.dtype, f'Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}'
    assert x_reshaped.shape[1] == weight.shape[1], f'Incompatible dimensions: {x_reshaped.shape} - {weight.shape}'
    assert bias is None or bias.shape[0] == weight.shape[0], 'Incompatible dimensions in between weight and bias'
    M, K = x_reshaped.shape
    N, K = weight.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_input = torch.empty_like(output) if save_act_input else None
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    kernel_fwd[grid](output, act_input, x_reshaped, weight, bias if bias is not None else x, M, N, K, M // 32, N // 32, K // 32, stride_cm=output.stride(0), stride_am=x_reshaped.stride(0), stride_ak=x_reshaped.stride(1), stride_bk=weight.stride(1), stride_bn=weight.stride(0), BIAS=bias is not None, SAVE_ACT_INPUT=save_act_input, ACTIVATION=activation, A_ROWMAJOR=x_reshaped.stride(1) == 1, B_COLMAJOR=weight.stride(1) == 1, GROUP_M=8)
    if not save_act_input:
        return output.reshape(*batch_shape, output.shape[-1])
    else:
        return (output.reshape(*batch_shape, output.shape[-1]), act_input.reshape(*batch_shape, act_input.shape[-1]))