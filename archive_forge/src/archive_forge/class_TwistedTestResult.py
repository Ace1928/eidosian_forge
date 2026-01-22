from collections import namedtuple
from testtools.tags import TagContext
class TwistedTestResult(LoggingBase):
    """
    Emulate the relevant bits of :py:class:`twisted.trial.itrial.IReporter`.

    Used to ensure that we can use ``trial`` as a test runner.
    """

    def __init__(self, event_log=None):
        super().__init__(event_log=event_log)
        self._was_successful = True
        self.testsRun = 0

    def startTest(self, test):
        self.testsRun += 1
        self._events.append(('startTest', test))

    def stopTest(self, test):
        self._events.append(('stopTest', test))

    def addSuccess(self, test):
        self._events.append(('addSuccess', test))

    def addError(self, test, error):
        self._was_successful = False
        self._events.append(('addError', test, error))

    def addFailure(self, test, error):
        self._was_successful = False
        self._events.append(('addFailure', test, error))

    def addExpectedFailure(self, test, failure, todo=None):
        self._events.append(('addExpectedFailure', test, failure))

    def addUnexpectedSuccess(self, test, todo=None):
        self._events.append(('addUnexpectedSuccess', test))

    def addSkip(self, test, reason):
        self._events.append(('addSkip', test, reason))

    def wasSuccessful(self):
        return self._was_successful

    def done(self):
        pass