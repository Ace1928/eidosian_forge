import numpy as np
from skimage._shared.testing import assert_equal
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import BRIEF, match_descriptors, corner_peaks, corner_harris
from skimage._shared import testing
Verify matched keypoints and their corresponding masks results between
    image and its rotated version with the expected keypoint pairs with
    cross_check enabled.