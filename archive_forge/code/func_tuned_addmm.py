import functools
import logging
from typing import Any, Dict, List, Optional
import torch
from torch._inductor.virtualized import V
from .. import config as inductor_config
from ..codegen.cuda.gemm_template import CUTLASSGemmTemplate
from ..lowering import register_lowering
from ..select_algorithm import (
from ..utils import (
from .mm_common import (
@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    ordered_kwargs_for_cpp_kernel = ('beta', 'alpha')
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    if m * n == 0 or not use_max_autotune():
        choices = [aten_addmm.bind((inp, mat1, mat2), layout, ordered_kwargs_for_cpp_kernel, alpha=alpha, beta=beta)] if use_aten_gemm_kernels() else []
        return autotune_select_algorithm('addmm', choices, [inp, mat1, mat2], layout)
    choices = [aten_addmm.bind((inp_expanded, mat1, mat2), layout, ordered_kwargs_for_cpp_kernel, alpha=alpha, beta=beta)] if use_aten_gemm_kernels() else []
    if use_aten_gemm_kernels() and inp_expanded.get_stride()[0] == 0 and (inp_expanded.get_device().type == 'cuda') and inductor_config.triton.autotune_cublasLt:
        choices.insert(0, aten_bias_addmm.bind((inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta))
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            mm_template.maybe_append_choice(choices, input_nodes=(inp_expanded, mat1, mat2), layout=layout, **mm_options(config, k, layout), prefix_args=1, epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta))
    if use_cutlass_template(layout):
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2, inp_expanded], alpha=alpha, beta=beta, input_reorder=[2, 0, 1], fuseable=False)
    return autotune_select_algorithm('addmm', choices, [inp_expanded, mat1, mat2], layout)