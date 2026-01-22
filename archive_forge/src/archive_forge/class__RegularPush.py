import logging
import re
import socket
import threading
import time
from timeit import default_timer
from typing import Callable, Tuple
from ..registry import CollectorRegistry, REGISTRY
class _RegularPush(threading.Thread):

    def __init__(self, pusher, interval, prefix):
        super().__init__()
        self._pusher = pusher
        self._interval = interval
        self._prefix = prefix

    def run(self):
        wait_until = default_timer()
        while True:
            while True:
                now = default_timer()
                if now >= wait_until:
                    while wait_until < now:
                        wait_until += self._interval
                    break
                time.sleep(wait_until - now)
            try:
                self._pusher.push(prefix=self._prefix)
            except OSError:
                logging.exception('Push failed')