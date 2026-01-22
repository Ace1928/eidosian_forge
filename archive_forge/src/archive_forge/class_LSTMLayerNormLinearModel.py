from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class LSTMLayerNormLinearModel(nn.Module):
    """Container module with an LSTM, a LayerNorm, and a linear."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, state = self.lstm(x)
        x = self.norm(x)
        x = self.linear(x)
        return (x, state)