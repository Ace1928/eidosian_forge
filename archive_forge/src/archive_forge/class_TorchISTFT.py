from typing import Optional
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
class TorchISTFT(nn.Module):
    """Multichannel Inverse-Short-Time-Fourier functional
    wrapper for torch.istft to support batches
    Args:
        STFT (Tensor): complex stft of
            shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            last axis is stacked real and imaginary
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        window (callable, optional): window function
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        length (int, optional): audio signal length to crop the signal
    Returns:
        x (Tensor): audio waveform of
            shape (nb_samples, nb_channels, nb_timesteps)
    """

    def __init__(self, n_fft: int=4096, n_hop: int=1024, center: bool=False, sample_rate: float=44100.0, window: Optional[nn.Parameter]=None) -> None:
        super(TorchISTFT, self).__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        self.sample_rate = sample_rate
        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

    def forward(self, X: Tensor, length: Optional[int]=None) -> Tensor:
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])
        y = torch.istft(torch.view_as_complex(X), n_fft=self.n_fft, hop_length=self.n_hop, window=self.window, center=self.center, normalized=False, onesided=True, length=length)
        y = y.reshape(shape[:-3] + y.shape[-1:])
        return y