from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class Conv2dPoolFlattenFunctional(nn.Module):
    """Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a functional Flatten followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=1), nn.ReLU(), nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True), nn.Tanh(), nn.AvgPool2d(kernel_size=2, stride=2, padding=1))
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(11, 13, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x