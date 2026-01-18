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
@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    choices = [aten_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    if m * n != 0 and use_triton_template(layout):
        for config in mm_configs(m, n, k):
            mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout))
    if m * n != 0 and use_cutlass_template(layout):
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True)
    from torch._inductor.ir import FixedLayout, FlexibleLayout
    if len(choices) == 1 and use_aten_gemm_kernels() and isinstance(layout, FixedLayout):
        layout = FlexibleLayout(device=layout.device, dtype=layout.dtype, size=layout.size)
        choices = [aten_mm.bind((mat1, mat2), layout)]
    return autotune_select_algorithm('mm', choices, [mat1, mat2], layout)