import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class InotifyBindingNotFoundError(PyinotifyError):
    """
    Raised when no inotify support couldn't be found.
    """

    def __init__(self):
        err = "Couldn't find any inotify binding"
        PyinotifyError.__init__(self, err)