import signal
from .utils import TimeoutException, BaseTimeout, base_timeoutable
class SignalTimeout(BaseTimeout):
    """Context manager for limiting in the time the execution of a block
    using signal.SIGALRM Unix signal.

    See :class:`stopit.utils.BaseTimeout` for more information
    """

    def __init__(self, seconds, swallow_exc=True):
        seconds = int(seconds)
        super(SignalTimeout, self).__init__(seconds, swallow_exc)

    def handle_timeout(self, signum, frame):
        self.state = BaseTimeout.TIMED_OUT
        raise TimeoutException('Block exceeded maximum timeout value (%d seconds).' % self.seconds)

    def setup_interrupt(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def suppress_interrupt(self):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)