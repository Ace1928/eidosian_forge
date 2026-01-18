from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_repeated_run_with_patches(self):

    def f():
        return (self.test_object.foo, self.test_object.bar, self.test_object.baz)
    self.monkey_patcher.add_patch(self.test_object, 'foo', 'haha')
    result = self.monkey_patcher.run_with_patches(f)
    self.assertEqual(('haha', self.original_object.bar, self.original_object.baz), result)
    result = self.monkey_patcher.run_with_patches(f)
    self.assertEqual(('haha', self.original_object.bar, self.original_object.baz), result)