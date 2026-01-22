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
def _guarded_task_generation(self, result_job, func, iterable):
    """Provides a generator of tasks for imap and imap_unordered with
        appropriate handling for iterables which throw exceptions during
        iteration."""
    try:
        i = -1
        for i, x in enumerate(iterable):
            yield (result_job, i, func, (x,), {})
    except Exception as e:
        yield (result_job, i + 1, _helper_reraises_exception, (e,), {})