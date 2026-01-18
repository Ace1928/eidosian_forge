import torch
from torch.nn.parameter import Parameter
from typing import List
@torch.jit.export
def toggle_observer_update(self, enabled=True):
    self.static_enabled[0] = int(enabled)
    return self