import collections
import errno
import heapq
import logging
import math
import os
import pyngus
import select
import socket
import threading
import time
import uuid
def _get_delay(self, max_delay=None):
    """Get the delay in milliseconds until the next callable needs to be
        run, or 'max_delay' if no outstanding callables or the delay to the
        next callable is > 'max_delay'.
        """
    due = self._deadlines[0] if self._deadlines else None
    if due is None:
        return max_delay
    _now = time.monotonic()
    if due <= _now:
        return 0
    else:
        return min(due - _now, max_delay) if max_delay else due - _now