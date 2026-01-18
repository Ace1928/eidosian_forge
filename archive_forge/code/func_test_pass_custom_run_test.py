from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test_pass_custom_run_test(self):

    class SomeCase(TestCase):

        def test_foo(self):
            pass
    result = TestResult()
    case = SomeCase('test_foo', runTest=CustomRunTest)
    from_run_test = case.run(result)
    self.assertThat(from_run_test, Is(CustomRunTest.marker))