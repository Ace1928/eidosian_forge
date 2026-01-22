import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback
import debugpy
from debugpy.common import json, timestamp, util
def _threads():
    output = '\n'.join([str(t) for t in threading.enumerate()])
    warning('$THREADS:\n\n{0}', output)