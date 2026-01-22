from typing import Callable, Optional
import torch
from torchaudio.prototype.functional import barkscale_fbanks, chroma_filterbank
from torchaudio.transforms import Spectrogram
class InverseBarkScale(torch.nn.Module):
    """Estimate a STFT in normal frequency domain from bark frequency domain.

    .. devices:: CPU CUDA

    It minimizes the euclidian norm between the input bark-spectrogram and the product between
    the estimated spectrogram and the filter banks using SGD.

    Args:
        n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_barks (int, optional): Number of bark filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        max_iter (int, optional): Maximum number of optimization iterations. (Default: ``100000``)
        tolerance_loss (float, optional): Value of loss to stop optimization at. (Default: ``1e-5``)
        tolerance_change (float, optional): Difference in losses to stop optimization at. (Default: ``1e-8``)
        sgdargs (dict or None, optional): Arguments for the SGD optimizer. (Default: ``None``)
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> mel_spectrogram_transform = transforms.BarkSpectrogram(sample_rate, n_fft=1024)
        >>> mel_spectrogram = bark_spectrogram_transform(waveform)
        >>> inverse_barkscale_transform = transforms.InverseBarkScale(n_stft=1024 // 2 + 1)
        >>> spectrogram = inverse_barkscale_transform(mel_spectrogram)
    """
    __constants__ = ['n_stft', 'n_barks', 'sample_rate', 'f_min', 'f_max', 'max_iter', 'tolerance_loss', 'tolerance_change', 'sgdargs']

    def __init__(self, n_stft: int, n_barks: int=128, sample_rate: int=16000, f_min: float=0.0, f_max: Optional[float]=None, max_iter: int=100000, tolerance_loss: float=1e-05, tolerance_change: float=1e-08, sgdargs: Optional[dict]=None, bark_scale: str='traunmuller') -> None:
        super(InverseBarkScale, self).__init__()
        self.n_barks = n_barks
        self.sample_rate = sample_rate
        self.f_max = f_max or float(sample_rate // 2)
        self.f_min = f_min
        self.max_iter = max_iter
        self.tolerance_loss = tolerance_loss
        self.tolerance_change = tolerance_change
        self.sgdargs = sgdargs or {'lr': 0.1, 'momentum': 0.9}
        if f_min > self.f_max:
            raise ValueError('Require f_min: {} <= f_max: {}'.format(f_min, self.f_max))
        fb = barkscale_fbanks(n_stft, self.f_min, self.f_max, self.n_barks, self.sample_rate, bark_scale)
        self.register_buffer('fb', fb)

    def forward(self, barkspec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            barkspec (torch.Tensor): A Bark frequency spectrogram of dimension (..., ``n_barks``, time)

        Returns:
            torch.Tensor: Linear scale spectrogram of size (..., freq, time)
        """
        shape = barkspec.size()
        barkspec = barkspec.view(-1, shape[-2], shape[-1])
        n_barks, time = (shape[-2], shape[-1])
        freq, _ = self.fb.size()
        barkspec = barkspec.transpose(-1, -2)
        if self.n_barks != n_barks:
            raise ValueError('Expected an input with {} bark bins. Found: {}'.format(self.n_barks, n_barks))
        specgram = torch.rand(barkspec.size()[0], time, freq, requires_grad=True, dtype=barkspec.dtype, device=barkspec.device)
        optim = torch.optim.SGD([specgram], **self.sgdargs)
        loss = float('inf')
        for _ in range(self.max_iter):
            optim.zero_grad()
            diff = barkspec - specgram.matmul(self.fb)
            new_loss = diff.pow(2).sum(axis=-1).mean()
            new_loss.backward()
            optim.step()
            specgram.data = specgram.data.clamp(min=0)
            new_loss = new_loss.item()
            if new_loss < self.tolerance_loss or abs(loss - new_loss) < self.tolerance_change:
                break
            loss = new_loss
        specgram.requires_grad_(False)
        specgram = specgram.clamp(min=0).transpose(-1, -2)
        specgram = specgram.view(shape[:-2] + (freq, time))
        return specgram