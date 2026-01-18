import functools
import torch
from ..lowering import lowerings
from ..select_algorithm import (
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import mm_args, mm_grid, mm_options
def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    """
    Computes mm(mat1, mat2) + mm(mat3, mat4)
    """
    m1, n1, k1, layout1, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    m2, n2, _, layout2, mat3, mat4 = mm_args(mat3, mat4, layout=layout)
    if m1 * n1 == 0 or m2 * n2 == 0 or (not V.graph.sizevars.statically_known_list_equals(mat1.get_size(), mat3.get_size())) or (not V.graph.sizevars.statically_known_list_equals(mat2.get_size(), mat4.get_size())):
        if m1 == m2 and n1 == n2:
            V.graph.sizevars.guard_equals(m1, m2)
            V.graph.sizevars.guard_equals(n1, n2)
            return lowerings[aten.addmm](lowerings[aten.mm](mat3, mat4), mat1, mat2)
        return lowerings[aten.add](lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4))
    assert layout1 == layout2
    choices = [aten_mm_plus_mm.bind((mat1, mat2, mat3, mat4), layout1)] if use_aten_gemm_kernels() else []
    if use_triton_template(layout1):
        for config in mm_configs():
            if config.kwargs['BLOCK_K'] < k1:
                mm_plus_mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2, mat3, mat4), layout=layout1, **mm_options(config, k1, layout1))
    return autotune_select_algorithm('mm_plus_mm', choices, [mat1, mat2, mat3, mat4], layout1)