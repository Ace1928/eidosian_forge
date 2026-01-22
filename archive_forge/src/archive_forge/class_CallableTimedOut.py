import functools
import logging
import os
import pipes
import shutil
import sys
import tempfile
import time
import unittest
from humanfriendly.compat import StringIO
from humanfriendly.text import random_string
class CallableTimedOut(Exception):
    """Raised by :func:`retry()` when the timeout expires."""