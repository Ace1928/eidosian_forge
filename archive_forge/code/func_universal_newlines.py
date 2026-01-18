import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
@universal_newlines.setter
def universal_newlines(self, universal_newlines):
    self.text_mode = bool(universal_newlines)