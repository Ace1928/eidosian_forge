import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
def _pipe_relay(self):
    while True:
        try:
            brk = False
            data = os.read(self._pipe_read_fd, 4096)
            if self._stopped.is_set():
                if _LAST_WRITE_TOKEN not in data:
                    n = len(_LAST_WRITE_TOKEN)
                    while n and data[-n:] != _LAST_WRITE_TOKEN[:n]:
                        n -= 1
                    if n:
                        data += os.read(self._pipe_read_fd, len(_LAST_WRITE_TOKEN) - n)
                if _LAST_WRITE_TOKEN in data:
                    data = data.replace(_LAST_WRITE_TOKEN, b'')
                    brk = True
            i = self._orig_src.write(data)
            if i is not None:
                while i < len(data):
                    i += self._orig_src.write(data[i:])
            self._queue.put(data)
            if brk:
                return
        except OSError:
            return