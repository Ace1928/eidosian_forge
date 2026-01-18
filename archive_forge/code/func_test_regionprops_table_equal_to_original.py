import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_regionprops_table_equal_to_original():
    regions = regionprops(SAMPLE, INTENSITY_FLOAT_SAMPLE)
    out_table = regionprops_table(SAMPLE, INTENSITY_FLOAT_SAMPLE, properties=COL_DTYPES.keys())
    for prop, dtype in COL_DTYPES.items():
        for i, reg in enumerate(regions):
            rp = reg[prop]
            if np.isscalar(rp) or prop in OBJECT_COLUMNS or dtype is np.object_:
                assert_array_equal(rp, out_table[prop][i])
            else:
                shape = rp.shape if isinstance(rp, np.ndarray) else (len(rp),)
                for ind in np.ndindex(shape):
                    modified_prop = '-'.join(map(str, (prop,) + ind))
                    loc = ind if len(ind) > 1 else ind[0]
                    assert_equal(rp[loc], out_table[modified_prop][i])