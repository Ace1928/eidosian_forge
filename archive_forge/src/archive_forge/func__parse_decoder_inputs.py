import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
def _parse_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
    """Prepares decoder inputs.

        Args:
            decoder_inputs (Tensor): Inputs used for teacher-forced training, i.e. mel-specs,
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``)

        Returns:
            inputs (Tensor): Processed decoder inputs with shape (max of ``mel_specgram_lengths``, n_batch, ``n_mels``).
        """
    decoder_inputs = decoder_inputs.transpose(1, 2)
    decoder_inputs = decoder_inputs.view(decoder_inputs.size(0), int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
    decoder_inputs = decoder_inputs.transpose(0, 1)
    return decoder_inputs