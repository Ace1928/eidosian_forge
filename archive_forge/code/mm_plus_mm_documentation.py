import functools
import torch
from ..lowering import lowerings
from ..select_algorithm import (
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import mm_args, mm_grid, mm_options

    Computes mm(mat1, mat2) + mm(mat3, mat4)
    