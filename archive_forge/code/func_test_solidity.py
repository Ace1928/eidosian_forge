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
def test_solidity():
    solidity = regionprops(SAMPLE)[0].solidity
    target_solidity = 0.576
    assert_almost_equal(solidity, target_solidity)
    solidity = regionprops(SAMPLE, spacing=(3, 9))[0].solidity
    assert_almost_equal(solidity, target_solidity)