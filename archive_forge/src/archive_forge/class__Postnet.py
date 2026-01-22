import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
class _Postnet(nn.Module):
    """Postnet Module.

    Args:
        n_mels (int): Number of mel bins.
        postnet_embedding_dim (int): Postnet embedding dimension.
        postnet_kernel_size (int): Postnet kernel size.
        postnet_n_convolution (int): Number of postnet convolutions.
    """

    def __init__(self, n_mels: int, postnet_embedding_dim: int, postnet_kernel_size: int, postnet_n_convolution: int):
        super().__init__()
        self.convolutions = nn.ModuleList()
        for i in range(postnet_n_convolution):
            in_channels = n_mels if i == 0 else postnet_embedding_dim
            out_channels = n_mels if i == postnet_n_convolution - 1 else postnet_embedding_dim
            init_gain = 'linear' if i == postnet_n_convolution - 1 else 'tanh'
            num_features = n_mels if i == postnet_n_convolution - 1 else postnet_embedding_dim
            self.convolutions.append(nn.Sequential(_get_conv1d_layer(in_channels, out_channels, kernel_size=postnet_kernel_size, stride=1, padding=int((postnet_kernel_size - 1) / 2), dilation=1, w_init_gain=init_gain), nn.BatchNorm1d(num_features)))
        self.n_convs = len(self.convolutions)

    def forward(self, x: Tensor) -> Tensor:
        """Pass the input through Postnet.

        Args:
            x (Tensor): The input sequence with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).

        Return:
            x (Tensor): Tensor with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
        """
        for i, conv in enumerate(self.convolutions):
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
        return x