import io
import os
import sys
from testtools import ExtendedToStreamDecorator
from testtools.run import (BUFFEROUTPUT, CATCHBREAK, FAILFAST, USAGE_AS_MAIN,
from subunit import StreamResultToBytes
from subunit.test_results import AutoTimingTestResultDecorator
class SubunitTestRunner(object):

    def __init__(self, verbosity=None, failfast=None, buffer=None, stream=None, stdout=None, tb_locals=False):
        """Create a TestToolsTestRunner.

        :param verbosity: Ignored.
        :param failfast: Stop running tests at the first failure.
        :param buffer: Ignored.
        :param stream: Upstream unittest stream parameter.
        :param stdout: Testtools stream parameter.
        :param tb_locals: Testtools traceback in locals parameter.

        Either stream or stdout can be supplied, and stream will take
        precedence.
        """
        self.failfast = failfast
        self.stream = stream or stdout or sys.stdout
        self.tb_locals = tb_locals

    def run(self, test):
        """Run the given test case or test suite."""
        result, _ = self._list(test)
        result = ExtendedToStreamDecorator(result)
        result = AutoTimingTestResultDecorator(result)
        if self.failfast is not None:
            result.failfast = self.failfast
            result.tb_locals = self.tb_locals
        result.startTestRun()
        try:
            test(result)
        finally:
            result.stopTestRun()
        return result

    def list(self, test, loader=None):
        """List the test."""
        result, errors = self._list(test)
        if loader is not None:
            errors = loader.errors
        if errors:
            failed_descr = '\n'.join(errors).encode('utf8')
            result.status(file_name='import errors', runnable=False, file_bytes=failed_descr, mime_type='text/plain;charset=utf8')
            sys.exit(2)

    def _list(self, test):
        test_ids, errors = list_test(test)
        try:
            fileno = self.stream.fileno()
        except:
            fileno = None
        if fileno is not None:
            stream = os.fdopen(fileno, 'wb', 0)
        else:
            stream = self.stream
        result = StreamResultToBytes(stream)
        for test_id in test_ids:
            result.status(test_id=test_id, test_status='exists')
        return (result, errors)