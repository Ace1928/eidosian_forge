import logging
from typing import List
from ..select_algorithm import autotune_select_algorithm, ChoiceCaller, TritonTemplate
from .mm_common import mm_args, mm_configs, mm_grid, mm_options
def tuned_uint4x2_mixed_mm(mat1, mat2, mat2_mm_shape, mat2_dtype):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=None, use_4x2_dim=True)
    choices: List[ChoiceCaller] = []
    b_prologue_cast_type = f'tl.{mat2_dtype}'.replace('torch.', '')
    for config in mm_configs(m, n, k):
        uint4x2_mixed_mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout, b_prologue_cast_type))
    return autotune_select_algorithm('uint4x2_mixed_mm', choices, [mat1, mat2], layout)