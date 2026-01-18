import torch
from ..lowering import register_lowering
from ..select_algorithm import (
from ..utils import ceildiv as cdiv, use_aten_gemm_kernels, use_triton_template
from .mm_common import addmm_epilogue, mm_args, mm_configs, mm_options
@register_lowering(aten.bmm)
def tuned_bmm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    choices = [aten_bmm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            bmm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout))
    return autotune_select_algorithm('bmm', choices, [mat1, mat2], layout)