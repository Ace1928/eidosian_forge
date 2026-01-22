import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class BilinearSeqAttn(nn.Module):
    """
    A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        self.xWy = xWy
        if self.training:
            alpha = F.log_softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha