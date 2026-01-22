import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO,
@property
def ETA(self):
    if self.done:
        prefix = 'Done'
        t = self.elapsed
    else:
        prefix = 'ETA '
        if self.max is None:
            t = -1
        elif self.elapsed == 0 or self.cur == self.min:
            t = 0
        else:
            t = float(self.max - self.min)
            t /= self.cur - self.min
            t = (t - 1) * self.elapsed
    return '%s: %s' % (prefix, self.format_duration(t))