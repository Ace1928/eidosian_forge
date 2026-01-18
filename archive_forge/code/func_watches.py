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
@property
def watches(self):
    """
        Get a reference on the internal watch manager dictionary.

        @return: Internal watch manager dictionary.
        @rtype: dict
        """
    return self._wmd