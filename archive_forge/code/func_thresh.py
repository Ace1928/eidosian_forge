import inspect
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..filters._multiotsu import (
from ..transform import integral_image
from ..util import dtype_limits
from ._sparse import _correlate_sparse, _validate_window_size
def thresh(func):
    """
        A wrapper function to return a thresholded image.
        """

    def wrapper(im):
        return im > func(im)
    try:
        wrapper.__orifunc__ = func.__orifunc__
    except AttributeError:
        wrapper.__orifunc__ = func.__module__ + '.' + func.__name__
    return wrapper