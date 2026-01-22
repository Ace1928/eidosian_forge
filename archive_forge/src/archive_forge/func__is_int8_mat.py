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
def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)