import os
import sys
import signal
import itertools
import logging
import threading
from _weakrefset import WeakSet
from multiprocessing import process as _mproc
def _set_current_process(process):
    global _current_process
    _current_process = _mproc._current_process = process