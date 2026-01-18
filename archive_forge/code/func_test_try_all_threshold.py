import math
import numpy as np
import pytest
from numpy.testing import (
from scipy import ndimage as ndi
from skimage import data, util
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.exposure import histogram
from skimage.filters._multiotsu import (
from skimage.filters.thresholding import (
@pytest.mark.skipif(not has_mpl, reason='matplotlib not installed')
def test_try_all_threshold(self):
    fig, ax = try_all_threshold(self.image)
    all_texts = [axis.texts for axis in ax if axis.texts != []]
    text_content = [text.get_text() for x in all_texts for text in x]
    assert 'RuntimeError' in text_content