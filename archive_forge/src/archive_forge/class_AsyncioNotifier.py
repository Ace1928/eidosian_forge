import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class AsyncioNotifier(Notifier):
    """

    asyncio/trollius event loop adapter.

    """

    def __init__(self, watch_manager, loop, callback=None, default_proc_fun=None, read_freq=0, threshold=0, timeout=None):
        """

        See examples/asyncio_notifier.py for an example usage.

        @param loop: asyncio or trollius event loop instance.
        @type loop: asyncio.BaseEventLoop or trollius.BaseEventLoop instance.
        @param callback: Functor called at the end of each call to handle_read.
                         Expects to receive the notifier object (self) as
                         single parameter.
        @type callback: callable object or function

        """
        self.loop = loop
        self.handle_read_callback = callback
        Notifier.__init__(self, watch_manager, default_proc_fun, read_freq, threshold, timeout)
        loop.add_reader(self._fd, self.handle_read)

    def stop(self):
        self.loop.remove_reader(self._fd)
        Notifier.stop(self)

    def handle_read(self, *args, **kwargs):
        self.read_events()
        self.process_events()
        if self.handle_read_callback is not None:
            self.handle_read_callback(self)