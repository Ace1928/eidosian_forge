import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
class UnknownDropout(nn.Module):
    """
    With set frequency, replaces tokens with unknown token.

    This layer can be used right before an embedding layer to make the model more robust
    to unknown words at test time.
    """

    def __init__(self, unknown_idx, probability):
        """
        Initialize layer.

        :param unknown_idx: index of unknown token, replace tokens with this
        :param probability: during training, replaces tokens with unknown token
                            at this rate.
        """
        super().__init__()
        self.unknown_idx = unknown_idx
        self.prob = probability

    def forward(self, input):
        """
        If training and dropout rate > 0, masks input with unknown token.
        """
        if self.training and self.prob > 0:
            mask = input.new(input.size()).float().uniform_(0, 1) < self.prob
            input.masked_fill_(mask, self.unknown_idx)
        return input