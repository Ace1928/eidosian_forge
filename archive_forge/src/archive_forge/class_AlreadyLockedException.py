from __future__ import print_function
import errno
import logging
import os
import time
from oauth2client import util
class AlreadyLockedException(Exception):
    """Trying to lock a file that has already been locked by the LockedFile."""
    pass