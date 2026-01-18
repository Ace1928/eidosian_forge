from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_uncertainty_nansum(self):
    uq = UncertainQuantity([1, 2], 'm', [1, 1])
    uq_nan = UncertainQuantity([1, 2, np.nan], 'm', [1, 1, np.nan])
    self.assertQuantityEqual(np.sum(uq), np.nansum(uq))
    self.assertQuantityEqual(np.sum(uq), uq_nan.nansum())