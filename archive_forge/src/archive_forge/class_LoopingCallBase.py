import logging
import sys
from eventlet import event
from eventlet import greenthread
from oslo_utils import timeutils
class LoopingCallBase(object):

    def __init__(self, f=None, *args, **kw):
        self.args = args
        self.kw = kw
        self.f = f
        self._running = False
        self.done = None

    def stop(self):
        self._running = False

    def wait(self):
        return self.done.wait()