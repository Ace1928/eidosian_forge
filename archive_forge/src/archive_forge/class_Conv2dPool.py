from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class Conv2dPool(nn.Module):
    """Model with only Conv2d layers, all with bias, some in a Sequential and some following.
    Activation function modules in between each layer, Pool2d modules in between each layer.
    Used to test pruned Conv2d-Pool2d-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=1), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True), nn.Tanh(), nn.AvgPool2d(kernel_size=2, stride=2, padding=1))
        self.conv2d1 = nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(48, 52, kernel_size=3, padding=1, bias=True)
        self.conv2d3 = nn.Conv2d(52, 52, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.maxpool(x)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = F.relu(x)
        x = self.conv2d3(x)
        return x