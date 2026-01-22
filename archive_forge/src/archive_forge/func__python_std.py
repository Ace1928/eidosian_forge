import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial
def _python_std(stream: str):
    return {'stdout': sys.stdout, 'stderr': sys.stderr}[stream]