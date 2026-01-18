import math
from itertools import product
from typing import cast, get_args, Literal
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.fft import fftshift
from scipy.stats import norm as normal_distribution  # type: ignore
from scipy.signal import get_window, welch, stft, istft, spectrogram
from scipy.signal._short_time_fft import FFT_MODE_TYPE, \
from scipy.signal.windows import gaussian
def test_scale_to():
    """Verify `scale_to()` method."""
    SFT = ShortTimeFFT(np.ones(4) * 2, hop=4, fs=1, scale_to=None)
    SFT.scale_to('magnitude')
    assert SFT.scaling == 'magnitude'
    assert SFT.fac_psd == 2.0
    assert SFT.fac_magnitude == 1
    SFT.scale_to('psd')
    assert SFT.scaling == 'psd'
    assert SFT.fac_psd == 1
    assert SFT.fac_magnitude == 0.5
    SFT.scale_to('psd')
    for scale, s_fac in zip(('magnitude', 'psd'), (8, 4)):
        SFT = ShortTimeFFT(np.ones(4) * 2, hop=4, fs=1, scale_to=None)
        dual_win = SFT.dual_win.copy()
        SFT.scale_to(cast(Literal['magnitude', 'psd'], scale))
        assert_allclose(SFT.dual_win, dual_win * s_fac)