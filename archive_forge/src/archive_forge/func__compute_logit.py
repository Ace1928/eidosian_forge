import torch
from torch import nn
def _compute_logit(self, hidden, weight, bias, proj):
    if proj is None:
        logit = nn.functional.linear(hidden, weight, bias=bias)
    else:
        proj_hid = nn.functional.linear(hidden, proj.t().contiguous())
        logit = nn.functional.linear(proj_hid, weight, bias=bias)
    return logit