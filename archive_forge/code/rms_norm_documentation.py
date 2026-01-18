import torch
from torch.nn import init
from flash_attn.ops.layer_norm import (
residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    