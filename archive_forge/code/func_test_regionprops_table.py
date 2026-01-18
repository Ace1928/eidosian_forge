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
def test_regionprops_table():
    out = regionprops_table(SAMPLE)
    assert out == {'label': np.array([1]), 'bbox-0': np.array([0]), 'bbox-1': np.array([0]), 'bbox-2': np.array([10]), 'bbox-3': np.array([18])}
    out = regionprops_table(SAMPLE, properties=('label', 'area', 'bbox'), separator='+')
    assert out == {'label': np.array([1]), 'area': np.array([72]), 'bbox+0': np.array([0]), 'bbox+1': np.array([0]), 'bbox+2': np.array([10]), 'bbox+3': np.array([18])}