import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
ensure cut_normalized returns the same output for the same input,
    when specifying random seed
    