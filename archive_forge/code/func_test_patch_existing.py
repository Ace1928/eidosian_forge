from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_patch_existing(self):
    self.monkey_patcher.add_patch(self.test_object, 'foo', 'haha')
    self.monkey_patcher.patch()
    self.assertEqual(self.test_object.foo, 'haha')