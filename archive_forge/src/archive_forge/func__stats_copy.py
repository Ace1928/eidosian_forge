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
def _stats_copy(self):
    self._stats_lock.acquire()
    try:
        return self._stats.copy()
    finally:
        self._stats_lock.release()