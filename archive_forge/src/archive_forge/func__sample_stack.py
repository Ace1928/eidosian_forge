import atexit
import os
import platform
import random
import sys
import threading
import time
import uuid
from collections import deque
import sentry_sdk
from sentry_sdk._compat import PY33, PY311
from sentry_sdk._lru_cache import LRUCache
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _sample_stack(*args, **kwargs):
    """
            Take a sample of the stack on all the threads in the process.
            This should be called at a regular interval to collect samples.
            """
    if not self.new_profiles and (not self.active_profiles):
        return
    new_profiles = len(self.new_profiles)
    now = nanosecond_time()
    try:
        sample = [(str(tid), extract_stack(frame, cache, cwd)) for tid, frame in sys._current_frames().items()]
    except AttributeError:
        capture_internal_exception(sys.exc_info())
        return
    for _ in range(new_profiles):
        self.active_profiles.add(self.new_profiles.popleft())
    inactive_profiles = []
    for profile in self.active_profiles:
        if profile.active:
            profile.write(now, sample)
        else:
            inactive_profiles.append(profile)
    for profile in inactive_profiles:
        self.active_profiles.remove(profile)