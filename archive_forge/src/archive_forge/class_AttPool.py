from typing import Tuple
import torch
import torch.nn as nn
import torchaudio
class AttPool(nn.Module):
    """Attention-Pooling module that estimates the attention score.

    Args:
        input_dim (int): Input feature dimension.
        att_dim (int): Attention Tensor dimension.
    """

    def __init__(self, input_dim: int, att_dim: int):
        super(AttPool, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1)
        self.linear2 = nn.Linear(input_dim, att_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and pooling.

        Args:
            x (torch.Tensor): Input Tensor with dimensions `(batch, time, feature_dim)`.

        Returns:
            (torch.Tensor): Attention score with dimensions `(batch, att_dim)`.
        """
        att = self.linear1(x)
        att = att.transpose(2, 1)
        att = nn.functional.softmax(att, dim=2)
        x = torch.matmul(att, x).squeeze(1)
        x = self.linear2(x)
        return x