from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_restore_non_existing(self):
    self.monkey_patcher.add_patch(self.test_object, 'doesntexist', 'value')
    self.monkey_patcher.patch()
    self.monkey_patcher.restore()
    marker = object()
    self.assertIs(marker, getattr(self.test_object, 'doesntexist', marker))