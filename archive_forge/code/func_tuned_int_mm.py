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
@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout, out_dtype=torch.int32)
    choices = [aten__int_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    if m * n != 0 and use_triton_template(layout, enable_int32=True):
        choices = []
        for config in int8_mm_configs(m, n, k):
            mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout))
    return autotune_select_algorithm('int_mm', choices, [mat1, mat2], layout)