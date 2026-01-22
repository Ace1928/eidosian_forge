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
class ChainIfTrue(ProcessEvent):
    """
    Makes conditional chaining depending on the result of the nested
    processing instance.
    """

    def my_init(self, func):
        """
        Method automatically called from base class constructor.
        """
        self._func = func

    def process_default(self, event):
        return not self._func(event)