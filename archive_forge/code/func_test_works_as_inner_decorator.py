from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test_works_as_inner_decorator(self):

    def wrapped(function):
        """Silly, trivial decorator."""

        def decorated(*args, **kwargs):
            return function(*args, **kwargs)
        decorated.__name__ = function.__name__
        decorated.__dict__.update(function.__dict__)
        return decorated

    class SomeCase(TestCase):

        @wrapped
        @run_test_with(CustomRunTest)
        def test_foo(self):
            pass
    result = TestResult()
    case = SomeCase('test_foo')
    from_run_test = case.run(result)
    self.assertThat(from_run_test, Is(CustomRunTest.marker))