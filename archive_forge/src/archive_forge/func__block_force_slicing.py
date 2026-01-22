import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def _block_force_slicing(arrays):
    arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
    return _block_slicing(arrays, list_ndim, result_ndim)