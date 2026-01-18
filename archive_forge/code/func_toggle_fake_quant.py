import torch
from torch.nn.parameter import Parameter
from typing import List
@torch.jit.export
def toggle_fake_quant(self, enabled=True):
    self.fake_quant_enabled[0] = int(enabled)
    return self