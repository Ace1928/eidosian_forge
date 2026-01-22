from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def _jacfwd(func, inputs, strict=False, vectorize=False):
    if strict:
        raise RuntimeError('torch.autograd.functional.jacobian: `strict=True` and `strategy="forward-mode"` are not supported together (yet). Please either set `strict=False` or `strategy="reverse-mode"`.')
    is_inputs_tuple, inputs = _as_tuple(inputs, 'inputs', 'jacobian')
    output_info = []
    if vectorize:
        input_numels = tuple((input.numel() for input in inputs))
        tangents = _construct_standard_basis_for(inputs, input_numels)

        def jvp(tangents):
            with fwAD.dual_level():
                dual_inputs = tuple((fwAD.make_dual(input, tangent.view_as(input)) for input, tangent in zip(inputs, tangents)))
                _is_outputs_tuple, dual_outputs = _as_tuple(func(*dual_inputs), 'outputs')
                output_info.append(_is_outputs_tuple)
                jv = []
                primal_outs = []
                for dual_out in dual_outputs:
                    primal, tangent = fwAD.unpack_dual(dual_out)
                    primal_outs.append(primal)
                    if tangent is not None:
                        jv.append(tangent)
                    else:
                        jv.append(torch.zeros_like(primal))
                output_info.append(primal_outs)
                return tuple(jv)
        outputs_before_split = _vmap(jvp)(tangents)
        is_outputs_tuple, outputs = output_info
        jacobian_input_output = []
        for jac_output_i, output_i in zip(outputs_before_split, outputs):
            jacobian_output_i_output = []
            for jac, input_j in zip(jac_output_i.split(input_numels, dim=0), inputs):
                jacobian_input_i_output_j = jac.permute(*range(1, jac.ndim), 0).reshape((*output_i.shape, *input_j.shape))
                jacobian_output_i_output.append(jacobian_input_i_output_j)
            jacobian_input_output.append(jacobian_output_i_output)
        return _tuple_postprocess(jacobian_input_output, (is_outputs_tuple, is_inputs_tuple))
    else:
        raise NotImplementedError('Computing Jacobian using forward-AD or forward-over-reverse Hessian isonly implemented for `vectorize=True`.')