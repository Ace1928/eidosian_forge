import torch
from ..lowering import register_lowering
from ..select_algorithm import (
from ..utils import ceildiv as cdiv, use_aten_gemm_kernels, use_triton_template
from .mm_common import addmm_epilogue, mm_args, mm_configs, mm_options
def tuned_baddbmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    m, n, k, layout, mat1, mat2, inp = mm_args(mat1, mat2, inp, layout=layout)
    choices = [aten_baddbmm.bind((inp, mat1, mat2), layout, alpha=alpha, beta=beta)] if use_aten_gemm_kernels() else []
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            bmm_template.maybe_append_choice(choices, input_nodes=(inp, mat1, mat2), layout=layout, **mm_options(config, k, layout), prefix_args=1, epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta))
    return autotune_select_algorithm('baddbmm', choices, [inp, mat1, mat2], layout)