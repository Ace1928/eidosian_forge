from contextlib import contextmanager
import datetime
import os
import re
import sys
import types
from collections import deque
from inspect import signature
from io import StringIO
from warnings import warn
from IPython.utils.decorators import undoc
from IPython.utils.py3compat import PYPY
from typing import Dict
def _break_one_group(self, group):
    while group.breakables:
        x = self.buffer.popleft()
        self.output_width = x.output(self.output, self.output_width)
        self.buffer_width -= x.width
    while self.buffer and isinstance(self.buffer[0], Text):
        x = self.buffer.popleft()
        self.output_width = x.output(self.output, self.output_width)
        self.buffer_width -= x.width