import string
from taskflow import test
from taskflow.utils import iter_utils
def test_unique_seen(self):
    iters = [['a', 'b'], ['a', 'c', 'd'], ['a', 'e', 'f'], ['f', 'm', 'n']]
    self.assertEqual(['a', 'b', 'c', 'd', 'e', 'f', 'm', 'n'], list(iter_utils.unique_seen(iters)))