from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_rescale(self):
    a = UncertainQuantity([1, 1, 1], pq.m, [0.1, 0.1, 0.1])
    b = a.rescale(pq.ft)
    self.assertQuantityEqual(a.rescale('ft'), [3.2808399, 3.2808399, 3.2808399] * pq.ft)
    self.assertQuantityEqual(a.rescale('ft').uncertainty, [0.32808399, 0.32808399, 0.32808399] * pq.ft)
    seventy_km = Quantity(70, pq.km, dtype=np.float32)
    seven_km = Quantity(7, pq.km, dtype=np.float32)
    seventyish_km = UncertainQuantity(seventy_km, pq.km, seven_km, dtype=np.float32)
    self.assertTrue(seventyish_km.dtype == np.float32)
    in_meters = seventyish_km.rescale(pq.m)
    self.assertTrue(in_meters.dtype == seventyish_km.dtype)
    seventyish_km_rescaled_idempotent = seventyish_km.rescale(pq.km)
    self.assertTrue(seventyish_km_rescaled_idempotent.dtype == np.float32)
    self.assertQuantityEqual(seventyish_km + in_meters, 2 * seventy_km)