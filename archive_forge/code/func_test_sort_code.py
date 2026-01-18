import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def test_sort_code(self):
    """Cannot sort by code object would need to get filename etc."""
    self.assertRaises(ValueError, self.stats.sort, 'code')