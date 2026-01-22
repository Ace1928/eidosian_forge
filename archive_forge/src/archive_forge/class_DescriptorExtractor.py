import numpy as np
from ..util import img_as_float
from .._shared.utils import (
class DescriptorExtractor:

    def __init__(self):
        self.descriptors_ = np.array([])

    def extract(self, image, keypoints):
        """Extract feature descriptors in image for given keypoints.

        Parameters
        ----------
        image : 2D array
            Input image.
        keypoints : (N, 2) array
            Keypoint locations as ``(row, col)``.

        """
        raise NotImplementedError()