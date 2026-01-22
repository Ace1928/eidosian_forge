import types, math, os, gc
import torch
from torch.nn import functional as F
@MyFunction
def FF_one(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
    xx = state[5 * i + 0].to(dtype=self.FLOAT_MODE)
    xk = x * time_mix_k + xx * (1 - time_mix_k)
    xr = x * time_mix_r + xx * (1 - time_mix_r)
    state[5 * i + 0] = x.float()
    r = torch.sigmoid(xr @ rw)
    k = torch.square(torch.relu(xk @ kw))
    kv = k @ vw
    return r * kv