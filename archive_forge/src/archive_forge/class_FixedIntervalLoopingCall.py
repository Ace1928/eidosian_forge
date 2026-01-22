import logging
import sys
from eventlet import event
from eventlet import greenthread
from oslo_utils import timeutils
class FixedIntervalLoopingCall(LoopingCallBase):
    """A fixed interval looping call."""

    def start(self, interval, initial_delay=None):
        self._running = True
        done = event.Event()

        def _inner():
            if initial_delay:
                greenthread.sleep(initial_delay)
            try:
                while self._running:
                    start = timeutils.utcnow()
                    self.f(*self.args, **self.kw)
                    end = timeutils.utcnow()
                    if not self._running:
                        break
                    delay = interval - timeutils.delta_seconds(start, end)
                    if delay <= 0:
                        LOG.warning('task run outlasted interval by %s sec', -delay)
                    greenthread.sleep(delay if delay > 0 else 0)
            except LoopingCallDone as e:
                self.stop()
                done.send(e.retvalue)
            except Exception:
                done.send_exception(*sys.exc_info())
                return
            else:
                done.send(True)
        self.done = done
        greenthread.spawn_n(_inner)
        return self.done