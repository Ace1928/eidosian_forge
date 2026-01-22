import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
class _Encoder(nn.Module):
    """Encoder Module.

    Args:
        encoder_embedding_dim (int): Number of embedding dimensions in the encoder.
        encoder_n_convolution (int): Number of convolution layers in the encoder.
        encoder_kernel_size (int): The kernel size in the encoder.

    Examples
        >>> encoder = _Encoder(3, 512, 5)
        >>> input = torch.rand(10, 20, 30)
        >>> output = encoder(input)  # shape: (10, 30, 512)
    """

    def __init__(self, encoder_embedding_dim: int, encoder_n_convolution: int, encoder_kernel_size: int) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList()
        for _ in range(encoder_n_convolution):
            conv_layer = nn.Sequential(_get_conv1d_layer(encoder_embedding_dim, encoder_embedding_dim, kernel_size=encoder_kernel_size, stride=1, padding=int((encoder_kernel_size - 1) / 2), dilation=1, w_init_gain='relu'), nn.BatchNorm1d(encoder_embedding_dim))
            self.convolutions.append(conv_layer)
        self.lstm = nn.LSTM(encoder_embedding_dim, int(encoder_embedding_dim / 2), 1, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters()

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        """Pass the input through the Encoder.

        Args:
            x (Tensor): The input sequences with shape (n_batch, encoder_embedding_dim, n_seq).
            input_lengths (Tensor): The length of each input sequence with shape (n_batch, ).

        Return:
            x (Tensor): A tensor with shape (n_batch, n_seq, encoder_embedding_dim).
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs