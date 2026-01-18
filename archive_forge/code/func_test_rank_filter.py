import inspect
import numpy as np
import pytest
from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as _3d_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte
@pytest.mark.parametrize('outdt', [None, np.float32, np.float64])
@pytest.mark.parametrize('filter', all_rank_filters)
def test_rank_filter(self, filter, outdt):

    @run_in_parallel(warnings_matching=['Possible precision loss'])
    def check():
        expected = self.refs[filter]
        if outdt is not None:
            out = np.zeros_like(expected, dtype=outdt)
        else:
            out = None
        result = getattr(rank, filter)(self.image, self.footprint, out=out)
        if filter == 'entropy':
            if outdt is not None:
                expected = expected.astype(outdt)
            assert_allclose(expected, result, atol=0, rtol=1e-15)
        elif filter == 'otsu':
            assert result[3, 5] in [41, 81]
            result[3, 5] = 81
            assert result[19, 18] in [141, 172]
            result[19, 18] = 172
            assert_array_almost_equal(expected, result)
        else:
            if outdt is not None:
                result = np.mod(result, 256.0).astype(expected.dtype)
            assert_array_almost_equal(expected, result)
    check()