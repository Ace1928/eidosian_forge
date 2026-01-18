import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def test_sort_totaltime(self):
    self.stats.sort('totaltime')
    code_list = [d.totaltime for d in self.stats.data]
    self.assertEqual(code_list, sorted(code_list, reverse=True))