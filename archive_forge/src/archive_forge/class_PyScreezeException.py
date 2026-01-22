import collections
import datetime
import functools
import os
import subprocess
import sys
import time
import errno
from contextlib import contextmanager
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import __version__ as PIL__version__
from PIL import ImageGrab
class PyScreezeException(Exception):
    """PyScreezeException is a generic exception class raised when a
    PyScreeze-related error happens. If a PyScreeze function raises an
    exception that isn't PyScreezeException or a subclass, assume it is
    a bug in PyScreeze."""
    pass