from functools import partial
from typing import Optional
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup
from flash_attn.ops.activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd
from flash_attn.utils.distributed import (

        process_group is required. We're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute pre_act and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            'auto': heuristic will be picked automatically:
                For CUDA >= 11.8, we set heuristic=0 for both fp16 and bf16 for best perf.
                For CUDA <= 11.7, we set heuristic=1 for fp16 and heuristic=-1 for bf16.
        