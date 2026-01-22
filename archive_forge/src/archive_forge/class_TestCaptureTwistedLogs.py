import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class TestCaptureTwistedLogs(NeedsTwistedTestCase):
    """Tests for CaptureTwistedLogs."""

    def test_captures_logs(self):
        from testtools.twistedsupport import CaptureTwistedLogs

        class SomeTest(TestCase):

            def test_something(self):
                self.useFixture(CaptureTwistedLogs())
                log.msg('foo')
        test = SomeTest('test_something')
        test.run()
        self.assertThat(test.getDetails(), MatchesDict({'twisted-log': AsText(Contains('foo'))}))