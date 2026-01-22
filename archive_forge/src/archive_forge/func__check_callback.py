import collections
import collections.abc
import concurrent.futures
import errno
import functools
import heapq
import itertools
import os
import socket
import stat
import subprocess
import threading
import time
import traceback
import sys
import warnings
import weakref
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import protocols
from . import sslproto
from . import staggered
from . import tasks
from . import transports
from . import trsock
from .log import logger
def _check_callback(self, callback, method):
    if coroutines.iscoroutine(callback) or coroutines.iscoroutinefunction(callback):
        raise TypeError(f'coroutines cannot be used with {method}()')
    if not callable(callback):
        raise TypeError(f'a callable object was expected by {method}(), got {callback!r}')