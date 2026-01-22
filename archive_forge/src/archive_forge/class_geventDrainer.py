import socket
import threading
import time
from collections import deque
from queue import Empty
from time import sleep
from weakref import WeakKeyDictionary
from kombu.utils.compat import detect_environment
from celery import states
from celery.exceptions import TimeoutError
from celery.utils.threads import THREAD_TIMEOUT_MAX
@register_drainer('gevent')
class geventDrainer(greenletDrainer):

    def spawn(self, func):
        import gevent
        g = gevent.spawn(func)
        gevent.sleep(0)
        return g

    def _create_drain_complete_event(self):
        from gevent.event import Event
        self._drain_complete_event = Event()

    def _send_drain_complete_event(self):
        self._drain_complete_event.set()
        self._create_drain_complete_event()