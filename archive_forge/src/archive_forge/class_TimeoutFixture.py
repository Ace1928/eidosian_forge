import itertools
from contextlib import ExitStack
class TimeoutFixture:
    """Kill a test with sigalarm if it runs too long.

    Only works on Unix at present.
    """

    def __init__(self, timeout_secs, gentle=True):
        import signal
        self.timeout_secs = timeout_secs
        self.alarm_fn = getattr(signal, 'alarm', None)
        self.gentle = gentle
        self._es = ExitStack()

    def signal_handler(self, signum, frame):
        raise TimeoutException(self.timeout_secs)

    def setUp(self):
        import signal
        if self.alarm_fn is None:
            return
        if self.gentle:
            old_handler = signal.signal(signal.SIGALRM, self.signal_handler)
        self._es.callback(self.alarm_fn, 0)
        self.alarm_fn(self.timeout_secs)
        if self.gentle:
            self._es.callback(signal.signal, signal.SIGALRM, old_handler)

    def cleanUp(self):
        self._es.close()