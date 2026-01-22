from typing import Callable, Optional
import torch
from torchaudio.prototype.functional import barkscale_fbanks, chroma_filterbank
from torchaudio.transforms import Spectrogram

        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Chromagram of size (..., ``n_chroma``, time).
        