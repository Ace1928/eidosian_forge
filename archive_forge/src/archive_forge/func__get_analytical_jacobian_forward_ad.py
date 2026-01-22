import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _get_analytical_jacobian_forward_ad(fn, inputs, outputs, *, check_grad_dtypes=False, all_u=None) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    """Compute the analytical Jacobian using forward mode AD of `fn(inputs)` using forward mode AD with respect to `target`.

    Return N * M Jacobians where N is the number of tensors in target that require grad and
    M is the number of non-integral outputs.
    Contrary to other functions here, this function requires "inputs" to actually be used by the function.
    The computed value is expected to be wrong if the function captures the inputs by side effect instead of
    using the passed ones (many torch.nn tests do this).

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        check_grad_dtypes: if True, will check that the gradient dtype are valid
        all_u (optional): if provided, the Jacobian will be right multiplied with this vector

    Returns:
        A tuple of M N-tuples of tensors
    """
    fwAD = torch.autograd.forward_ad
    tensor_inputs = tuple((i for i in inputs if is_tensor_like(i) and i.requires_grad))
    if any((i.is_complex() for i in tensor_inputs)):
        raise ValueError('Expected inputs to be non-complex for _get_analytical_jacobian_forward_ad.')
    if all_u:
        jacobians = tuple((_allocate_jacobians_with_outputs(outputs, 1) for i in tensor_inputs))
    else:
        jacobians = tuple((_allocate_jacobians_with_outputs(outputs, i.numel()) for i in tensor_inputs))
    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        for i, inp in enumerate(inputs):
            if is_tensor_like(inp) and inp.requires_grad:
                if inp.layout == torch._mkldnn:
                    raise ValueError('MKLDNN inputs are not support for forward AD gradcheck.')
                inp = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
                fw_grads.append(fwAD.unpack_dual(inp)[1])
            dual_inputs.append(inp)
        if all_u:
            for i, (fw_grad, u) in enumerate(zip(fw_grads, all_u)):
                fw_grad.copy_(u.view_as(fw_grad))
                raw_outputs = _as_tuple(fn(*dual_inputs))
                dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                for index_o, d_o in enumerate(dual_outputs):
                    val, res = fwAD.unpack_dual(d_o)
                    if check_grad_dtypes and res is not None and (val.is_complex() != res.is_complex()):
                        raise GradcheckError('Forward AD gradient has dtype mismatch.')
                    jacobians[i][index_o].squeeze_(0)
                    if res is None:
                        jacobians[i][index_o].zero_()
                    else:
                        jacobians[i][index_o].copy_(res.reshape(-1))
                fw_grad.zero_()
        else:
            for i, fw_grad in enumerate(fw_grads):
                for lin_idx, grad_idx in enumerate(product(*[range(m) for m in fw_grad.size()])):
                    fw_grad[grad_idx] = 1.0
                    raw_outputs = _as_tuple(fn(*dual_inputs))
                    dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                    for index_o, d_o in enumerate(dual_outputs):
                        val, res = fwAD.unpack_dual(d_o)
                        if check_grad_dtypes and res is not None and (val.is_complex() != res.is_complex()):
                            raise GradcheckError('Forward AD gradient has dtype mismatch.')
                        if res is None:
                            jacobians[i][index_o][lin_idx].zero_()
                        else:
                            jacobians[i][index_o][lin_idx].copy_(res.reshape(-1))
                    fw_grad[grad_idx] = 0.0
    return jacobians