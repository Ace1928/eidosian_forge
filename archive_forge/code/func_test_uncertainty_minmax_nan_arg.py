from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_uncertainty_minmax_nan_arg(self):
    q = [[1, 2], [3, 4]] * pq.m
    self.assertQuantityEqual(q.min(), 1 * pq.m)
    self.assertQuantityEqual(q.max(), 4 * pq.m)
    self.assertQuantityEqual(q.argmin(), 0)
    self.assertQuantityEqual(q.argmax(), 3)
    uq = UncertainQuantity([[1, 2], [3, 4]], pq.m, [[1, 1], [1, 1]])
    self.assertQuantityEqual(uq.min(), 1 * pq.m)
    self.assertQuantityEqual(uq.max(), 4 * pq.m)
    self.assertQuantityEqual(uq.argmin(), 0)
    self.assertQuantityEqual(uq.argmax(), 3)
    nanq = [[1, 2], [3, 4], [np.nan, np.nan]] * pq.m
    nanuq = UncertainQuantity([[1, 2], [3, 4], [np.nan, np.nan]], pq.m, [[1, 1], [1, 1], [np.nan, np.nan]])
    self.assertQuantityEqual(nanq.nanmin(), 1 * pq.m)
    self.assertQuantityEqual(nanq.nanmax(), 4 * pq.m)
    self.assertQuantityEqual(nanq.nanargmin(), 0)
    self.assertQuantityEqual(nanq.nanargmax(), 3)
    self.assertQuantityEqual(nanuq.nanmin(), 1 * pq.m)
    self.assertQuantityEqual(nanuq.nanmax(), 4 * pq.m)
    self.assertQuantityEqual(nanuq.nanargmin(), 0)
    self.assertQuantityEqual(nanuq.nanargmax(), 3)