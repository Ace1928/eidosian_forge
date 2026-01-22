import threading
from collections import deque
from time import time
from sentry_sdk._types import TYPE_CHECKING
def _qsize(self):
    return len(self.queue)