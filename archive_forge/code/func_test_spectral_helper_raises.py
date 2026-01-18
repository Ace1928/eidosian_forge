from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_spectral_helper_raises(self):
    for kwargs in [{'y': self.y + 1, 'mode': 'complex'}, {'y': self.y + 1, 'mode': 'magnitude'}, {'y': self.y + 1, 'mode': 'angle'}, {'y': self.y + 1, 'mode': 'phase'}, {'mode': 'spam'}, {'y': self.y, 'sides': 'eggs'}, {'y': self.y, 'NFFT': 10, 'noverlap': 20}, {'NFFT': 10, 'noverlap': 10}, {'y': self.y, 'NFFT': 10, 'window': np.ones(9)}]:
        with pytest.raises(ValueError):
            mlab._spectral_helper(x=self.y, **kwargs)