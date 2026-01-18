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
def test_regionprops_table_deprecated_scalar_property():
    out = regionprops_table(SAMPLE, properties=('bbox_area',))
    assert list(out.keys()) == ['bbox_area']