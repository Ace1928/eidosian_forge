import numpy as np
from scipy import linalg
from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
def _flipudlr(array):
    """Reverse the rows and columns of an array."""
    return np.flipud(np.fliplr(array))