import threading
from collections import deque
from time import time
from sentry_sdk._types import TYPE_CHECKING
class FullError(Exception):
    """Exception raised by Queue.put(block=0)/put_nowait()."""
    pass