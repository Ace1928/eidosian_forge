import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_str_with_unicode(self):
    u = '§'
    matcher = EndsWith(u)
    self.assertEqual(f'EndsWith({u!r})', str(matcher))