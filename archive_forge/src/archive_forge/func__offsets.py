import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def _offsets(grad, hess):
    """Compute position refinement offsets from gradient and Hessian.

    This is equivalent to np.linalg.solve(-H, J) where H is the Hessian
    matrix and J is the gradient (Jacobian).

    This analytical solution is adapted from (BSD-licensed) C code by
    Otero et. al (see SIFT docstring References).
    """
    h00, h11, h22, h01, h02, h12 = hess
    g0, g1, g2 = grad
    det = h00 * h11 * h22
    det -= h00 * h12 * h12
    det -= h01 * h01 * h22
    det += 2 * h01 * h02 * h12
    det -= h02 * h02 * h11
    aa = (h11 * h22 - h12 * h12) / det
    ab = (h02 * h12 - h01 * h22) / det
    ac = (h01 * h12 - h02 * h11) / det
    bb = (h00 * h22 - h02 * h02) / det
    bc = (h01 * h02 - h00 * h12) / det
    cc = (h00 * h11 - h01 * h01) / det
    offset0 = -aa * g0 - ab * g1 - ac * g2
    offset1 = -ab * g0 - bb * g1 - bc * g2
    offset2 = -ac * g0 - bc * g1 - cc * g2
    return np.stack((offset0, offset1, offset2), axis=-1)