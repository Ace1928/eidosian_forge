from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_psd_windowarray(self):
    freqs = self.freqs_density
    spec, fsp = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=np.ones(self.NFFT_density_real))
    assert_allclose(fsp, freqs, atol=1e-06)
    assert spec.shape == freqs.shape