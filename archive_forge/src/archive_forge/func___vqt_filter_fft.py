import warnings
import numpy as np
from numba import jit
from . import audio
from .intervals import interval_frequencies
from .fft import get_fftlib
from .convert import cqt_frequencies, note_to_hz
from .spectrum import stft, istft
from .pitch import estimate_tuning
from .._cache import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError
from numpy.typing import DTypeLike
from typing import Optional, Union, Collection, List
from .._typing import _WindowSpec, _PadMode, _FloatLike_co, _ensure_not_reachable
@cache(level=10)
def __vqt_filter_fft(sr, freqs, filter_scale, norm, sparsity, hop_length=None, window='hann', gamma=0.0, dtype=np.complex64, alpha=None):
    """Generate the frequency domain variable-Q filter basis."""
    basis, lengths = filters.wavelet(freqs=freqs, sr=sr, filter_scale=filter_scale, norm=norm, pad_fft=True, window=window, gamma=gamma, alpha=alpha)
    n_fft = basis.shape[1]
    if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))
    basis *= lengths[:, np.newaxis] / float(n_fft)
    fft = get_fftlib()
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, :n_fft // 2 + 1]
    fft_basis = util.sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)
    return (fft_basis, n_fft, lengths)