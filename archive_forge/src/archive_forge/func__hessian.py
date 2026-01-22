import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def _hessian(d, positions):
    """Compute the non-redundant 3D Hessian terms at the requested positions.

    Source: "Anatomy of the SIFT Method"  p.380 (13)
    """
    p0 = positions[..., 0]
    p1 = positions[..., 1]
    p2 = positions[..., 2]
    two_d0 = 2 * d[p0, p1, p2]
    h00 = d[p0 - 1, p1, p2] + d[p0 + 1, p1, p2] - two_d0
    h11 = d[p0, p1 - 1, p2] + d[p0, p1 + 1, p2] - two_d0
    h22 = d[p0, p1, p2 - 1] + d[p0, p1, p2 + 1] - two_d0
    h01 = 0.25 * (d[p0 + 1, p1 + 1, p2] - d[p0 - 1, p1 + 1, p2] - d[p0 + 1, p1 - 1, p2] + d[p0 - 1, p1 - 1, p2])
    h02 = 0.25 * (d[p0 + 1, p1, p2 + 1] - d[p0 + 1, p1, p2 - 1] + d[p0 - 1, p1, p2 - 1] - d[p0 - 1, p1, p2 + 1])
    h12 = 0.25 * (d[p0, p1 + 1, p2 + 1] - d[p0, p1 + 1, p2 - 1] + d[p0, p1 - 1, p2 - 1] - d[p0, p1 - 1, p2 + 1])
    return (h00, h11, h22, h01, h02, h12)