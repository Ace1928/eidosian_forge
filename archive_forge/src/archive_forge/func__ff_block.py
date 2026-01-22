import math
import torch
import torch.nn as nn
from fairscale.nn.moe.moe_layer import MOELayer
from fairscale.nn.moe.top2gate import Top2Gate
def _ff_block(self, x):
    if self.is_moe:
        return self.moe_layer(x)
    else:
        return self.ff_block(x)