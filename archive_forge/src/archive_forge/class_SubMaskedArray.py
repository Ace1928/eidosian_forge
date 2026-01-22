import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
class SubMaskedArray(MaskedArray):
    """Pure subclass of MaskedArray, keeping some info on subclass."""

    def __new__(cls, info=None, **kwargs):
        obj = super().__new__(cls, **kwargs)
        obj._optinfo['info'] = info
        return obj