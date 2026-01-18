import pickle
import copy
from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase
 A few pickles collected before fixing #113 just to make sure we remain backwards compatible. 