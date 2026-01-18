import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def test_save_to_pickle(self):
    path = self._temppath('pkl')
    self.stats.save(path)
    with open(path, 'rb') as f:
        data1 = pickle.load(f)
        self.assertEqual(type(data1), lsprof.Stats)