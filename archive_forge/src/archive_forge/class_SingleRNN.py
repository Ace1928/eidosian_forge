import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class SingleRNN(nn.Module):

    def __init__(self, rnn_type: str, input_size: int, hidden_size: int, dropout: float=0.0) -> None:
        super(SingleRNN, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn: nn.modules.Module = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.proj(out)
        return out