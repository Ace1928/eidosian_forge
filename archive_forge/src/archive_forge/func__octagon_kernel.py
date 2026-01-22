import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, convolve
from ..transform import integral_image
from .corner import structure_tensor
from ..morphology import octagon, star
from .censure_cy import _censure_dob_loop
from ..feature.util import (
from .._shared.utils import check_nD
def _octagon_kernel(mo, no, mi, ni):
    outer = (mo + 2 * no) ** 2 - 2 * no * (no + 1)
    inner = (mi + 2 * ni) ** 2 - 2 * ni * (ni + 1)
    outer_weight = 1.0 / (outer - inner)
    inner_weight = 1.0 / inner
    c = (mo + 2 * no - (mi + 2 * ni)) // 2
    outer_oct = octagon(mo, no)
    inner_oct = np.zeros((mo + 2 * no, mo + 2 * no))
    inner_oct[c:-c, c:-c] = octagon(mi, ni)
    bfilter = outer_weight * outer_oct - (outer_weight + inner_weight) * inner_oct
    return bfilter