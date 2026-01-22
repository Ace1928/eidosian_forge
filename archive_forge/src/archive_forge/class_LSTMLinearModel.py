from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class LSTMLinearModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a linear."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        output, hidden = self.lstm(input)
        decoded = self.linear(output)
        return (decoded, output)