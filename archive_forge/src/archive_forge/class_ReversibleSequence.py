from typing import List
import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from xformers.components import RequiresWrappedInputs
class ReversibleSequence(nn.Module):

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])

    def forward(self, x, arg_route=(True, False), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        return _ReversibleFunction.apply(x, self.blocks, block_kwargs)