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
def _save_input(self, input):
    if self.stdin and self._input is None:
        self._input_offset = 0
        self._input = input
        if input is not None and self.text_mode:
            self._input = self._input.encode(self.stdin.encoding, self.stdin.errors)