import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
Using Tacotron2 for inference. The input is a batch of encoded
        sentences (``tokens``) and its corresponding lengths (``lengths``). The
        output is the generated mel spectrograms, its corresponding lengths, and
        the attention weights from the decoder.

        The input `tokens` should be padded with zeros to length max of ``lengths``.

        Args:
            tokens (Tensor): The input tokens to Tacotron2 with shape `(n_batch, max of lengths)`.
            lengths (Tensor or None, optional):
                The valid length of each sample in ``tokens`` with shape `(n_batch, )`.
                If ``None``, it is assumed that the all the tokens are valid. Default: ``None``

        Returns:
            (Tensor, Tensor, Tensor):
                Tensor
                    The predicted mel spectrogram with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    The length of the predicted mel spectrogram with shape `(n_batch, )`.
                Tensor
                    Sequence of attention weights from the decoder with shape
                    `(n_batch, max of mel_specgram_lengths, max of lengths)`.
        