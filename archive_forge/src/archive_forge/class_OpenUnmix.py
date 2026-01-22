from typing import Optional, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from .filtering import wiener
from .transforms import make_filterbanks, ComplexNorm
class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(self, nb_bins: int=4096, nb_channels: int=2, hidden_size: int=512, nb_layers: int=3, unidirectional: bool=False, input_mean: Optional[np.ndarray]=None, input_scale: Optional[np.ndarray]=None, max_bin: Optional[int]=None):
        super(OpenUnmix, self).__init__()
        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins
        self.hidden_size = hidden_size
        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = BatchNorm1d(hidden_size)
        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2
        self.lstm = LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=nb_layers, bidirectional=not unidirectional, batch_first=False, dropout=0.4 if nb_layers > 1 else 0)
        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)
        self.bn2 = BatchNorm1d(hidden_size)
        self.fc3 = Linear(in_features=hidden_size, out_features=self.nb_output_bins * nb_channels, bias=False)
        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)
        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[:self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)
        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[:self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)
        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)
        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        mix = x.detach().clone()
        x = x[..., :self.nb_bins]
        x = x + self.input_mean
        x = x * self.input_scale
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)
        lstm_out = self.lstm(x)
        x = torch.cat([x, lstm_out[0]], -1)
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x *= self.output_scale
        x += self.output_mean
        x = F.relu(x) * mix
        return x.permute(1, 2, 3, 0)