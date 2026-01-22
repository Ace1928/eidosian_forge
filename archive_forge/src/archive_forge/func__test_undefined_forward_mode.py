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
def _test_undefined_forward_mode(func, outputs, inputs):
    fwAD = torch.autograd.forward_ad
    inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    all_v, all_u, all_u_dense = _make_vectors(inp_tensors, outputs, use_forward_ad=True)
    tensor_inputs = tuple((i for i in inputs if is_tensor_like(i) and i.requires_grad))
    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        tensor_indices = set()
        for i, inp in enumerate(inputs):
            if is_tensor_like(inp) and inp.requires_grad:
                if inp.layout == torch._mkldnn:
                    raise ValueError('MKLDNN inputs are not support for forward AD gradcheck.')
                inp = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
                fw_grads.append(fwAD.unpack_dual(inp)[1])
                tensor_indices.add(i)
            dual_inputs.append(inp)
        for i, (fw_grad, u) in enumerate(zip(fw_grads, all_u)):
            fw_grad.copy_(u.view_as(fw_grad))
        for idx, inp in enumerate(inputs):
            if idx not in tensor_indices:
                continue
            dual_inp_obj = dual_inputs[idx]
            dual_inputs[idx] = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs1 = filter(_is_float_or_complex_tensor, raw_outputs)
            dual_inputs[idx] = inp.detach()
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs2 = filter(_is_float_or_complex_tensor, raw_outputs)
            dual_inputs[idx] = dual_inp_obj
            for index_o, (d_o1, d_o2) in enumerate(zip(dual_outputs1, dual_outputs2)):
                val1, res1 = fwAD.unpack_dual(d_o1)
                val2, res2 = fwAD.unpack_dual(d_o2)
                if not (res1 is None or res2 is None):
                    if not torch.allclose(res1, res2):
                        raise GradcheckError('Mismatch in tangent values for output with index: ', index_o, ' when input: ', inp, ' has an undefined tangent value. ', ' Got: ', res1, ' but expected: ', res2)
    return True