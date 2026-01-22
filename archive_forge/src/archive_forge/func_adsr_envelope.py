import warnings
from typing import List, Optional, Union
import torch
from torchaudio.functional import fftconvolve
def adsr_envelope(num_frames: int, *, attack: float=0.0, hold: float=0.0, decay: float=0.0, sustain: float=1.0, release: float=0.0, n_decay: int=2, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None):
    """Generate ADSR Envelope

    .. devices:: CPU CUDA

    Args:
        num_frames (int): The number of output frames.
        attack (float, optional):
            The relative *time* it takes to reach the maximum level from
            the start. (Default: ``0.0``)
        hold (float, optional):
            The relative *time* the maximum level is held before
            it starts to decay. (Default: ``0.0``)
        decay (float, optional):
            The relative *time* it takes to sustain from
            the maximum level. (Default: ``0.0``)
        sustain (float, optional): The relative *level* at which
            the sound should sustain. (Default: ``1.0``)

            .. Note::
               The duration of sustain is derived as `1.0 - (The sum of attack, hold, decay and release)`.

        release (float, optional): The relative *time* it takes for the sound level to
            reach zero after the sustain. (Default: ``0.0``)
        n_decay (int, optional): The degree of polynomial decay. Default: ``2``.
        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default
            (see :py:func:`torch.set_default_tensor_type`).
        device (torch.device, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :py:func:`torch.set_default_tensor_type`).
            device will be the CPU for CPU tensor types and the current CUDA
            device for CUDA tensor types.

    Returns:
        Tensor: ADSR Envelope. Shape: `(num_frames, )`

    Example
        .. image:: https://download.pytorch.org/torchaudio/doc-assets/adsr_examples.png

    """
    if not 0 <= attack <= 1:
        raise ValueError(f'The value of `attack` must be within [0, 1]. Found: {attack}')
    if not 0 <= decay <= 1:
        raise ValueError(f'The value of `decay` must be within [0, 1]. Found: {decay}')
    if not 0 <= sustain <= 1:
        raise ValueError(f'The value of `sustain` must be within [0, 1]. Found: {sustain}')
    if not 0 <= hold <= 1:
        raise ValueError(f'The value of `hold` must be within [0, 1]. Found: {hold}')
    if not 0 <= release <= 1:
        raise ValueError(f'The value of `release` must be within [0, 1]. Found: {release}')
    if attack + decay + release + hold > 1:
        raise ValueError('The sum of `attack`, `hold`, `decay` and `release` must not exceed 1.')
    nframes = num_frames - 1
    num_a = int(nframes * attack)
    num_h = int(nframes * hold)
    num_d = int(nframes * decay)
    num_r = int(nframes * release)
    out = torch.full((num_frames,), float(sustain), device=device, dtype=dtype)
    if num_a > 0:
        torch.linspace(0.0, 1.0, num_a + 1, out=out[:num_a + 1])
    if num_h > 0:
        out[num_a:num_a + num_h + 1] = 1.0
    if num_d > 0:
        i = num_a + num_h
        decay = out[i:i + num_d + 1]
        torch.linspace(1.0, 0.0, num_d + 1, out=decay)
        decay **= n_decay
        decay *= 1.0 - sustain
        decay += sustain
    if num_r > 0:
        torch.linspace(sustain, 0, num_r + 1, out=out[-num_r - 1:])
    return out