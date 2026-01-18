import string
from taskflow import test
from taskflow.utils import iter_utils
def test_unique_seen_empty(self):
    iters = []
    self.assertEqual([], list(iter_utils.unique_seen(iters)))