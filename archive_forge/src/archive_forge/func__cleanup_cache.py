import datetime as dt
import functools
import hashlib
import inspect
import io
import os
import pathlib
import pickle
import sys
import threading
import time
import unittest
import unittest.mock
import weakref
from contextlib import contextmanager
import param
from param.parameterized import iscoroutinefunction
from .state import state
def _cleanup_cache(cache, policy, max_items, time):
    """
    Deletes items in the cache if the exceed the number of items or
    their TTL (time-to-live) has expired.
    """
    while len(cache) >= max_items:
        if policy.lower() == 'fifo':
            key = list(cache.keys())[0]
        elif policy.lower() == 'lru':
            key = sorted(((k, time - t) for k, (_, _, _, t) in cache.items()), key=lambda o: o[1])[-1][0]
        elif policy.lower() == 'lfu':
            key = sorted(cache.items(), key=lambda o: o[1][2])[0][0]
        del cache[key]