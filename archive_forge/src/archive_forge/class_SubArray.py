import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
class SubArray(np.ndarray):

    def __new__(cls, arr, info={}):
        x = np.asanyarray(arr).view(cls)
        x.info = info.copy()
        return x

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.info = getattr(obj, 'info', {}).copy()
        return

    def __add__(self, other):
        result = super().__add__(other)
        result.info['added'] = result.info.get('added', 0) + 1
        return result

    def __iadd__(self, other):
        result = super().__iadd__(other)
        result.info['iadded'] = result.info.get('iadded', 0) + 1
        return result