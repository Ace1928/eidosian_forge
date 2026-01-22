import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def amplitude_to_DB(x: Tensor, multiplier: float, amin: float, db_multiplier: float, top_db: Optional[float]=None) -> Tensor:
    """Turn a spectrogram from the power/amplitude scale to the decibel scale.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    The output of each tensor in a batch depends on the maximum value of that tensor,
    and so may return different values for an audio clip split into snippets vs. a full clip.

    Args:

        x (Tensor): Input spectrogram(s) before being converted to decibel scale.
            The expected shapes are ``(freq, time)``, ``(channel, freq, time)`` or
            ``(..., batch, channel, freq, time)``.

            .. note::

               When ``top_db`` is specified, cut-off values are computed for each audio
               in the batch. Therefore if the input shape is 4D (or larger), different
               cut-off values are used for audio data in the batch.
               If the input shape is 2D or 3D, a single cutoff value is used.

        multiplier (float): Use 10. for power and 20. for amplitude
        amin (float): Number to clamp ``x``
        db_multiplier (float): Log10(max(reference value and amin))
        top_db (float or None, optional): Minimum negative cut-off in decibels. A reasonable number
            is 80. (Default: ``None``)

    Returns:
        Tensor: Output tensor in decibel scale
    """
    x_db = multiplier * torch.log10(torch.clamp(x, min=amin))
    x_db -= multiplier * db_multiplier
    if top_db is not None:
        shape = x_db.size()
        packed_channels = shape[-3] if x_db.dim() > 2 else 1
        x_db = x_db.reshape(-1, packed_channels, shape[-2], shape[-1])
        x_db = torch.max(x_db, (x_db.amax(dim=(-3, -2, -1)) - top_db).view(-1, 1, 1, 1))
        x_db = x_db.reshape(shape)
    return x_db