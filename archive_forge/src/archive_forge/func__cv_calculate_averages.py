import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_calculate_averages(image, Hphi):
    """Returns the average values 'inside' and 'outside'."""
    H = Hphi
    Hinv = 1.0 - H
    Hsum = np.sum(H)
    Hinvsum = np.sum(Hinv)
    avg_inside = np.sum(image * H)
    avg_oustide = np.sum(image * Hinv)
    if Hsum != 0:
        avg_inside /= Hsum
    if Hinvsum != 0:
        avg_oustide /= Hinvsum
    return (avg_inside, avg_oustide)