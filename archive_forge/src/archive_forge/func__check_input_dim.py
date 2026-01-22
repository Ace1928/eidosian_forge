import torch
import torch.ao.nn.intrinsic as nni
def _check_input_dim(self, input):
    if len(input.shape) != 5:
        raise ValueError('Input shape must be `(N, C, H, W)`!')