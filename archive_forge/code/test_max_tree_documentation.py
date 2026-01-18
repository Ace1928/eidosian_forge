import numpy as np
from skimage.morphology import max_tree, area_closing, area_opening
from skimage.morphology import max_tree_local_maxima, diameter_opening
from skimage.morphology import diameter_closing
from skimage.util import invert
from skimage._shared.testing import assert_array_equal, TestCase
tests the detection of maxima in 3D.