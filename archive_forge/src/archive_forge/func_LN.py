import types, math, os, gc
import torch
from torch.nn import functional as F
def LN(self, x, w):
    return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)