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
@pytest.mark.parametrize('filter', all_rank_filters)
def test_rank_filter_footprint_sequence_unsupported(self, filter):
    footprint_sequence = morphology.diamond(3, decomposition='sequence')
    with pytest.raises(ValueError):
        getattr(rank, filter)(self.image.astype(np.uint8), footprint_sequence)