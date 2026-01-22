import os
import sys
from eventlet import patcher, support
from eventlet.hubs import hub
def _init_kqueue(self):
    self.kqueue = select.kqueue()
    self._pid = os.getpid()