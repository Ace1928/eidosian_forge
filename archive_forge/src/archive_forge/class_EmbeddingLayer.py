import math
import torch
import torch.nn as nn
from fairscale.nn.moe.moe_layer import MOELayer
from fairscale.nn.moe.top2gate import Top2Gate
class EmbeddingLayer(nn.Embedding):
    """Wrapped nn.Embedding layer to allow for weight initialization."""

    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp_sqrt = math.sqrt(ninp)
        self.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return super().forward(src) * self.ninp_sqrt