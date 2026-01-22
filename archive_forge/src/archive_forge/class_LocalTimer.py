import traceback
import eventlet.hubs
from eventlet.support import greenlets as greenlet
import io
class LocalTimer(Timer):

    def __init__(self, *args, **kwargs):
        self.greenlet = greenlet.getcurrent()
        Timer.__init__(self, *args, **kwargs)

    @property
    def pending(self):
        if self.greenlet is None or self.greenlet.dead:
            return False
        return not self.called

    def __call__(self, *args):
        if not self.called:
            self.called = True
            if self.greenlet is not None and self.greenlet.dead:
                return
            cb, args, kw = self.tpl
            cb(*args, **kw)

    def cancel(self):
        self.greenlet = None
        Timer.cancel(self)