from .. import units as pq
from .common import TestCase
def test_str_format_scalar(self):
    self._check(1 * pq.J, '1.0 J')