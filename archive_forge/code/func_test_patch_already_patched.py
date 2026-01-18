from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_patch_already_patched(self):
    self.monkey_patcher.add_patch(self.test_object, 'foo', 'blah')
    self.monkey_patcher.add_patch(self.test_object, 'foo', 'BLAH')
    self.monkey_patcher.patch()
    self.assertEqual(self.test_object.foo, 'BLAH')
    self.monkey_patcher.restore()
    self.assertEqual(self.test_object.foo, self.original_object.foo)