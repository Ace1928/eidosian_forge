from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
class _FakeReactor:

    def __init__(self):
        self._running = False
        self._clock = task.Clock()
        self.callLater = self._clock.callLater
        self.seconds = self._clock.seconds
        self.getDelayedCalls = self._clock.getDelayedCalls
        self._whenRunning = []
        self._shutdownTriggers = {'before': [], 'during': []}

    def callWhenRunning(self, callable, *args, **kwargs):
        if self._whenRunning is None:
            callable(*args, **kwargs)
        else:
            self._whenRunning.append((callable, args, kwargs))

    def addSystemEventTrigger(self, phase, event, callable, *args):
        assert phase in ('before', 'during')
        assert event == 'shutdown'
        self._shutdownTriggers[phase].append((callable, args))

    def run(self):
        """
        Call timed events until there are no more or the reactor is stopped.

        @raise RuntimeError: When no timed events are left and the reactor is
            still running.
        """
        self._running = True
        whenRunning = self._whenRunning
        self._whenRunning = None
        for callable, args, kwargs in whenRunning:
            callable(*args, **kwargs)
        while self._running:
            calls = self.getDelayedCalls()
            if not calls:
                raise RuntimeError('No DelayedCalls left')
            self._clock.advance(calls[0].getTime() - self.seconds())
        shutdownTriggers = self._shutdownTriggers
        self._shutdownTriggers = None
        for trigger, args in shutdownTriggers['before'] + shutdownTriggers['during']:
            trigger(*args)

    def stop(self):
        """
        Stop the reactor.
        """
        if not self._running:
            raise error.ReactorNotRunning()
        self._running = False