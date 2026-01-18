import pickle
import copy
from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase
def test_uncertainquantity_object_dtype(self):
    x = UncertainQuantity(20, 'm', 0.2, dtype=object)
    y = pickle.loads(pickle.dumps(x))
    self.assertQuantityEqual(x, y)