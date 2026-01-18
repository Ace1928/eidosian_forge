import numpy as np
from ..feature.util import (
from .corner import corner_fast, corner_orientations, corner_peaks, corner_harris
from ..transform import pyramid_gaussian
from .._shared.utils import check_nD
from .._shared.compat import NP_COPY_IF_NEEDED
from .orb_cy import _orb_loop
Detect oriented FAST keypoints and extract rBRIEF descriptors.

        Note that this is faster than first calling `detect` and then
        `extract`.

        Parameters
        ----------
        image : 2D array
            Input image.

        