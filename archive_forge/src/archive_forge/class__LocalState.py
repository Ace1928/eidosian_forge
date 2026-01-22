import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
class _LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).
    Also a failed experiments with trying to provide some frequency based attention.
    """

    def __init__(self, channels: int, heads: int=4, ndecay: int=4):
        """
        Args:
            channels (int): Size of Conv1d layers.
            heads (int, optional):  (default: 4)
            ndecay (int, optional): (default: 4)
        """
        super(_LocalState, self).__init__()
        if channels % heads != 0:
            raise ValueError('Channels must be divisible by heads.')
        self.heads = heads
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
        if ndecay:
            self.query_decay.weight.data *= 0.01
            if self.query_decay.bias is None:
                raise ValueError('bias must not be None.')
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * 0, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LocalState forward call

        Args:
            x (torch.Tensor): input tensor for LocalState

        Returns:
            Tensor
                Output after being run through LocalState layer.
        """
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        delta = indexes[:, None] - indexes[None, :]
        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        dots = torch.einsum('bhct,bhcs->bhts', keys, queries)
        dots /= math.sqrt(keys.shape[2])
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = -decays.view(-1, 1, 1) * delta.abs() / math.sqrt(self.ndecay)
            dots += torch.einsum('fts,bhfs->bhts', decay_kernel, decay_q)
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)
        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum('bhts,bhct->bhcs', weights, content)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)