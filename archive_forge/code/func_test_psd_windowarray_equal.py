from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_psd_windowarray_equal(self):
    win = mlab.window_hanning(np.ones(self.NFFT_density_real))
    speca, fspa = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=win)
    specb, fspb = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides)
    assert_array_equal(fspa, fspb)
    assert_allclose(speca, specb, atol=1e-08)