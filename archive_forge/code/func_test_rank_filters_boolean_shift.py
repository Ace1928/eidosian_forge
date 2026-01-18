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
@pytest.mark.parametrize('shift_name', ['shift_x', 'shift_y'])
@pytest.mark.parametrize('shift_value', [False, True])
def test_rank_filters_boolean_shift(self, filter, shift_name, shift_value):
    """Test warning if shift is provided as a boolean."""
    filter_func = getattr(rank, filter)
    image = img_as_ubyte(self.image)
    kwargs = {'footprint': self.footprint, shift_name: shift_value}
    with pytest.warns() as record:
        filter_func(image, **kwargs)
        expected_lineno = inspect.currentframe().f_lineno - 1
    assert len(record) == 1
    assert 'will be interpreted as int' in record[0].message.args[0]
    assert record[0].filename == __file__
    assert record[0].lineno == expected_lineno