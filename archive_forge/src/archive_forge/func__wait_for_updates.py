import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
def _wait_for_updates(self, sentinels, change_notifier, timeout):
    time.sleep(timeout)