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
def test_props_to_dict():
    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions)
    assert out == {'label': np.array([1]), 'bbox-0': np.array([0]), 'bbox-1': np.array([0]), 'bbox-2': np.array([10]), 'bbox-3': np.array([18])}
    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions, properties=('label', 'area', 'bbox'), separator='+')
    assert out == {'label': np.array([1]), 'area': np.array([72]), 'bbox+0': np.array([0]), 'bbox+1': np.array([0]), 'bbox+2': np.array([10]), 'bbox+3': np.array([18])}