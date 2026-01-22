from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class LinearActivationFunctional(nn.Module):
    """Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and functional
    activationals are called in between each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(7, 5, bias=True), nn.ReLU(), nn.Linear(5, 6, bias=False), nn.ReLU(), nn.Linear(6, 4, bias=True))
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.linear2 = nn.Linear(3, 8, bias=False)
        self.linear3 = nn.Linear(8, 10, bias=False)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.seq(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return x