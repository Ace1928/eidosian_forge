import unittest
from traits.api import (
@cached_property
def _get_sum(self):
    self.pcalls += 1
    r = self.ref
    return r.int1 + r.int2 + r.int3