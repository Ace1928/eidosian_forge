import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized
class BiasHook:

    def __init__(self, parametrization, prune_bias):
        self.param = parametrization
        self.prune_bias = prune_bias

    def __call__(self, module, input, output):
        if getattr(module, '_bias', None) is not None:
            bias = module._bias.data
            if self.prune_bias:
                bias[~self.param.mask] = 0
            idx = [1] * len(output.shape)
            idx[1] = -1
            bias = bias.reshape(idx)
            output += bias
        return output