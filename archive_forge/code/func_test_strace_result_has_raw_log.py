import threading
from breezy import strace, tests
from breezy.strace import StraceResult, strace_detailed
from breezy.tests.features import strace_feature
def test_strace_result_has_raw_log(self):
    """Checks that a reasonable raw strace log was found by strace."""
    self._check_threads()

    def function():
        self.build_tree(['myfile'])
    unused, result = self.strace_detailed_or_skip(function, [], {}, follow_children=False)
    self.assertContainsRe(result.raw_log, 'myfile')