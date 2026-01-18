import dropout_layer_norm
import torch
from torch.nn import init
residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    