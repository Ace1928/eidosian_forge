import sys
from testtools.testresult import ExtendedToOriginalDecorator
def _run_one(self, result):
    """Run one test reporting to result.

        :param result: A testtools.TestResult to report activity to.
            This result object is decorated with an ExtendedToOriginalDecorator
            to ensure that the latest TestResult API can be used with
            confidence by client code.
        :return: The result object the test was run against.
        """
    return self._run_prepared_result(ExtendedToOriginalDecorator(result))