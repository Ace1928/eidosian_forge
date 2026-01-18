import threading
from breezy import strace, tests
from breezy.strace import StraceResult, strace_detailed
from breezy.tests.features import strace_feature
def test_strace_callable_result(self):
    self._check_threads()

    def function():
        return 'foo'
    result, strace_result = self.strace_detailed_or_skip(function, [], {}, follow_children=False)
    self.assertEqual('foo', result)
    self.assertIsInstance(strace_result, StraceResult)