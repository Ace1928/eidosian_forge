from .. import units as pq
from .common import TestCase
def test_str_format_non_scalar(self):
    self._check([1, 2] * pq.J, '[1. 2.] J')