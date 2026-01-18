import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)