import threading
from breezy import strace, tests
from breezy.strace import StraceResult, strace_detailed
from breezy.tests.features import strace_feature
class TestStrace(tests.TestCaseWithTransport):
    _test_needs_features = [strace_feature]

    def setUp(self):
        raise tests.TestSkipped('strace selftests are broken and disabled')

    def _check_threads(self):
        active = threading.activeCount()
        if active > 1:
            self.knownFailure('%d active threads, bug #103133 needs to be fixed.' % active)

    def strace_detailed_or_skip(self, *args, **kwargs):
        """Run strace, but cope if it's not allowed"""
        try:
            return strace_detailed(*args, **kwargs)
        except strace.StraceError as e:
            if e.err_messages.startswith('attach: ptrace(PTRACE_ATTACH, ...): Operation not permitted'):
                raise tests.TestSkipped('ptrace not permitted')
            else:
                raise

    def test_strace_callable_is_called(self):
        self._check_threads()
        output = []

        def function(positional, *args, **kwargs):
            output.append((positional, args, kwargs))
        self.strace_detailed_or_skip(function, ['a', 'b'], {'c': 'c'}, follow_children=False)
        self.assertEqual([('a', ('b',), {'c': 'c'})], output)

    def test_strace_callable_result(self):
        self._check_threads()

        def function():
            return 'foo'
        result, strace_result = self.strace_detailed_or_skip(function, [], {}, follow_children=False)
        self.assertEqual('foo', result)
        self.assertIsInstance(strace_result, StraceResult)

    def test_strace_result_has_raw_log(self):
        """Checks that a reasonable raw strace log was found by strace."""
        self._check_threads()

        def function():
            self.build_tree(['myfile'])
        unused, result = self.strace_detailed_or_skip(function, [], {}, follow_children=False)
        self.assertContainsRe(result.raw_log, 'myfile')