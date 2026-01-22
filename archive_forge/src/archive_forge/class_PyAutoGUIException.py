from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
class PyAutoGUIException(Exception):
    """
    PyAutoGUI code will raise this exception class for any invalid actions. If PyAutoGUI raises some other exception,
    you should assume that this is caused by a bug in PyAutoGUI itself. (Including a failure to catch potential
    exceptions raised by PyAutoGUI.)
    """
    pass