import pytest
from thinc.api import (
from thinc.compat import has_tensorflow, has_torch
class PyTorchModel(torch.nn.Module):

    def __init__(self, width, nO, nI, dropout):
        super(PyTorchModel, self).__init__()
        self.dropout1 = torch.nn.Dropout2d(dropout)
        self.dropout2 = torch.nn.Dropout2d(dropout)
        self.fc1 = torch.nn.Linear(nI, width)
        self.fc2 = torch.nn.Linear(width, nO)

    def forward(self, x):
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output