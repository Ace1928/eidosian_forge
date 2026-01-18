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
def tuned_mixed_mm(mat1, mat2, mat2_dtype):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=None)
    choices = [aten_fallback_mixed_mm.bind((mat1, mat2), layout)]
    if mat1.layout.dtype != torch.float32 and (not mat2.layout.is_contiguous()) or _is_sm7x_or_older_gpu(layout.device.index):
        return autotune_select_algorithm('mixed_mm', choices, [mat1, mat2], layout)
    if inductor_config.force_mixed_mm:
        choices = []
    b_prologue_cast_type = f'tl.{mat2_dtype}'.replace('torch.', '')
    has_int8_tensor = _is_int8_mat(mat1) or _is_int8_mat(mat2)
    for config in mm_configs(m, n, k, has_int8_tensor=has_int8_tensor):
        mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout, b_prologue_cast_type))
    return autotune_select_algorithm('mixed_mm', choices, [mat1, mat2], layout)