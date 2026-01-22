import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def _wait_timers(self):
    for x in self.timers:
        try:
            x.wait()
        except eventlet.greenlet.GreenletExit:
            pass
        except Exception:
            LOG.exception('Error waiting on timer.')