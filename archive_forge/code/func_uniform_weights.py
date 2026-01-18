import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def uniform_weights(x, x_mask):
    """
    Return uniform weights over non-masked input.
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha