import copy
import numpy as np
from .._shared.filters import gaussian
from .._shared.utils import check_nD
from .brief_cy import _brief_loop
from .util import (
Extract BRIEF binary descriptors for given keypoints in image.

        Parameters
        ----------
        image : 2D array
            Input image.
        keypoints : (N, 2) array
            Keypoint coordinates as ``(row, col)``.

        