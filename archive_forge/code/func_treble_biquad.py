import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def treble_biquad(waveform: Tensor, sample_rate: int, gain: float, central_freq: float=3000, Q: float=0.707) -> Tensor:
    """Design a treble tone-control effect.  Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        gain (float or torch.Tensor): desired gain at the boost (or attenuation) in dB.
        central_freq (float or torch.Tensor, optional): central frequency (in Hz). (Default: ``3000``)
        Q (float or torch.Tensor, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``).

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    Reference:
        - http://sox.sourceforge.net/sox.html
        - https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    dtype = waveform.dtype
    device = waveform.device
    central_freq = torch.as_tensor(central_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    gain = torch.as_tensor(gain, dtype=dtype, device=device)
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q
    A = torch.exp(gain / 40 * math.log(10))
    temp1 = 2 * torch.sqrt(A) * alpha
    temp2 = (A - 1) * torch.cos(w0)
    temp3 = (A + 1) * torch.cos(w0)
    b0 = A * (A + 1 + temp2 + temp1)
    b1 = -2 * A * (A - 1 + temp3)
    b2 = A * (A + 1 + temp2 - temp1)
    a0 = A + 1 - temp2 + temp1
    a1 = 2 * (A - 1 - temp3)
    a2 = A + 1 - temp2 - temp1
    return biquad(waveform, b0, b1, b2, a0, a1, a2)