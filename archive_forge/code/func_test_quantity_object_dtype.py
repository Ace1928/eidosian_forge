import pickle
import copy
from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase
def test_quantity_object_dtype(self):
    x = Quantity(1, dtype=object)
    y = pickle.loads(pickle.dumps(x))
    self.assertQuantityEqual(x, y)