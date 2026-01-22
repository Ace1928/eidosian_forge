import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
def _on_blocking(signum, frame):
    import inspect
    raise RuntimeError(f'Blocking detection timed-out at: {inspect.getframeinfo(frame)}')