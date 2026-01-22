import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
def _parse_decoder_outputs(self, mel_specgram: Tensor, gate_outputs: Tensor, alignments: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Prepares decoder outputs for output

        Args:
            mel_specgram (Tensor): mel spectrogram with shape (max of ``mel_specgram_lengths``, n_batch, ``n_mels``)
            gate_outputs (Tensor): predicted stop token with shape (max of ``mel_specgram_lengths``, n_batch)
            alignments (Tensor): sequence of attention weights from the decoder
                with shape (max of ``mel_specgram_lengths``, n_batch, max of ``text_lengths``)

        Returns:
            mel_specgram (Tensor): mel spectrogram with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``)
            gate_outputs (Tensor): predicted stop token with shape (n_batch, max of ``mel_specgram_lengths``)
            alignments (Tensor): sequence of attention weights from the decoder
                with shape (n_batch, max of ``mel_specgram_lengths``, max of ``text_lengths``)
        """
    alignments = alignments.transpose(0, 1).contiguous()
    gate_outputs = gate_outputs.transpose(0, 1).contiguous()
    mel_specgram = mel_specgram.transpose(0, 1).contiguous()
    shape = (mel_specgram.shape[0], -1, self.n_mels)
    mel_specgram = mel_specgram.view(*shape)
    mel_specgram = mel_specgram.transpose(1, 2)
    return (mel_specgram, gate_outputs, alignments)