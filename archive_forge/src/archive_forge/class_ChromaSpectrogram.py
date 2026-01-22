from typing import Callable, Optional
import torch
from torchaudio.prototype.functional import barkscale_fbanks, chroma_filterbank
from torchaudio.transforms import Spectrogram
class ChromaSpectrogram(torch.nn.Module):
    """Generates chromagram for audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd

    Composes :py:func:`torchaudio.transforms.Spectrogram` and
    and :py:func:`torchaudio.prototype.transforms.ChromaScale`.

    Args:
        sample_rate (int): Sample rate of audio signal.
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins.
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., torch.Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \\times \\text{hop\\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        n_chroma (int, optional): Number of chroma. (Default: ``12``)
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.ChromaSpectrogram(sample_rate=sample_rate, n_fft=400)
        >>> chromagram = transform(waveform)  # (channel, n_chroma, time)
    """

    def __init__(self, sample_rate: int, n_fft: int, *, win_length: Optional[int]=None, hop_length: Optional[int]=None, pad: int=0, window_fn: Callable[..., torch.Tensor]=torch.hann_window, power: float=2.0, normalized: bool=False, wkwargs: Optional[dict]=None, center: bool=True, pad_mode: str='reflect', n_chroma: int=12, tuning: float=0.0, ctroct: float=5.0, octwidth: Optional[float]=2.0, norm: int=2, base_c: bool=True):
        super().__init__()
        self.spectrogram = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, pad=pad, window_fn=window_fn, power=power, normalized=normalized, wkwargs=wkwargs, center=center, pad_mode=pad_mode, onesided=True)
        self.chroma_scale = ChromaScale(sample_rate, n_fft // 2 + 1, n_chroma=n_chroma, tuning=tuning, base_c=base_c, ctroct=ctroct, octwidth=octwidth, norm=norm)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Chromagram of size (..., ``n_chroma``, time).
        """
        spectrogram = self.spectrogram(waveform)
        chroma_spectrogram = self.chroma_scale(spectrogram)
        return chroma_spectrogram