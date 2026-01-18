from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
Tests the fix for a bug producing a label < start_label (gh-6240).

    For the v0.19.1 release, the `img` and `slic` call as below result in two
    non-contiguous regions with value 0 despite `start_label=1`. We verify that
    the minimum label is now `start_label` as expected.
    