import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
def _starts(self, bsz):
    """
        Return bsz start tokens.
        """
    return self.START.detach().expand(bsz, 1)