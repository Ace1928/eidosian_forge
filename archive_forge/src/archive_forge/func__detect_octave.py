import numpy as np
from ..feature.util import (
from .corner import corner_fast, corner_orientations, corner_peaks, corner_harris
from ..transform import pyramid_gaussian
from .._shared.utils import check_nD
from .._shared.compat import NP_COPY_IF_NEEDED
from .orb_cy import _orb_loop
def _detect_octave(self, octave_image):
    dtype = octave_image.dtype
    fast_response = corner_fast(octave_image, self.fast_n, self.fast_threshold)
    keypoints = corner_peaks(fast_response, min_distance=1)
    if len(keypoints) == 0:
        return (np.zeros((0, 2), dtype=dtype), np.zeros((0,), dtype=dtype), np.zeros((0,), dtype=dtype))
    mask = _mask_border_keypoints(octave_image.shape, keypoints, distance=16)
    keypoints = keypoints[mask]
    orientations = corner_orientations(octave_image, keypoints, OFAST_MASK)
    harris_response = corner_harris(octave_image, method='k', k=self.harris_k)
    responses = harris_response[keypoints[:, 0], keypoints[:, 1]]
    return (keypoints, orientations, responses)