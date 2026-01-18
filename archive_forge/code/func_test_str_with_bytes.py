import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_str_with_bytes(self):
    b = _b('§')
    matcher = EndsWith(b)
    self.assertEqual(f'EndsWith({b!r})', str(matcher))