import pickle
import copy
from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase
def test_unitconstant_persistence(self):
    x = constants.m_e
    y = pickle.loads(pickle.dumps(x))
    self.assertQuantityEqual(x, y)