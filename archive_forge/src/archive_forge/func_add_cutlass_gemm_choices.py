import copy
import logging
import re
from typing import cast, Dict, List, Optional, Tuple
from ...config import cuda as inductor_cuda_config
from ...ir import Buffer, CUDATemplateBuffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer
from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
from .cutlass_epilogue_gen import (
@staticmethod
def add_cutlass_gemm_choices(choices, layout, input_nodes, alpha=1, beta=0, input_reorder=None, fuseable=True, non_fuseable=True):
    if non_fuseable:
        if fuseable:
            can_fuse_epilogue = False
        else:
            can_fuse_epilogue = None
        cutlass_template = CUTLASSGemmTemplate(input_nodes, layout, alpha=alpha, beta=beta, input_reorder=input_reorder, can_fuse_epilogue=can_fuse_epilogue)
        ops = cutlass_template.gen_ops()
        for op in ops:
            cutlass_template.maybe_append_choice(choices, op=op)
    else:
        ops = []
    if fuseable:
        cutlass_template_evt = CUTLASSGemmTemplate(input_nodes, layout, alpha=alpha, beta=beta, input_reorder=input_reorder, can_fuse_epilogue=True)
        ops_evt = cutlass_template_evt.gen_ops()
        for op in ops_evt:
            cutlass_template_evt.maybe_append_choice(choices, op=op)
    else:
        ops_evt = []
    log.debug('Added %d cutlass gemm configs and %d fuseable gemm configs.', len(ops), len(ops_evt))