import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def _create_scalespace(self, image):
    """Source: "Anatomy of the SIFT Method" Alg. 1
        Construction of the scalespace by gradually blurring (scales) and
        downscaling (octaves) the image.
        """
    scalespace = []
    if self.upsampling > 1:
        image = rescale(image, self.upsampling, order=1)
    image = gaussian(image, sigma=self.upsampling * math.sqrt(self.sigma_min ** 2 - self.sigma_in ** 2), mode='reflect')
    tmp = np.power(2, np.arange(self.n_scales + 3) / self.n_scales)
    tmp *= self.sigma_min
    sigmas = self.deltas[:, np.newaxis] / self.deltas[0] * tmp[np.newaxis, :]
    self.scalespace_sigmas = sigmas
    var_diff = np.diff(sigmas * sigmas, axis=1)
    gaussian_sigmas = np.sqrt(var_diff) / self.deltas[:, np.newaxis]
    for o in range(self.n_octaves):
        octave = np.empty((self.n_scales + 3,) + image.shape, dtype=self.float_dtype, order='C')
        octave[0] = image
        for s in range(1, self.n_scales + 3):
            gaussian(octave[s - 1], sigma=gaussian_sigmas[o, s - 1], mode='reflect', out=octave[s])
        scalespace.append(np.moveaxis(octave, 0, -1))
        if o < self.n_octaves - 1:
            image = octave[self.n_scales][::2, ::2]
    return scalespace