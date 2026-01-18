import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_pickle_nested(self):
    st = self.module.StaticTuple('foo', self.module.StaticTuple('bar'))
    pickled = pickle.dumps(st)
    unpickled = pickle.loads(pickled)
    self.assertEqual(unpickled, st)