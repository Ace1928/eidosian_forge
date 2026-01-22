from collections import namedtuple
from testtools.tags import TagContext
class Python26TestResult(LoggingBase):
    """A precisely python 2.6 like test result, that logs."""

    def __init__(self, event_log=None):
        super().__init__(event_log=event_log)
        self.shouldStop = False
        self._was_successful = True
        self.testsRun = 0

    def addError(self, test, err):
        self._was_successful = False
        self._events.append(('addError', test, err))

    def addFailure(self, test, err):
        self._was_successful = False
        self._events.append(('addFailure', test, err))

    def addSuccess(self, test):
        self._events.append(('addSuccess', test))

    def startTest(self, test):
        self._events.append(('startTest', test))
        self.testsRun += 1

    def stop(self):
        self.shouldStop = True

    def stopTest(self, test):
        self._events.append(('stopTest', test))

    def wasSuccessful(self):
        return self._was_successful