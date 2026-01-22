from __future__ import division
import math
import os
import signal
import sys
import time
from .compat import *  # for: any, next
from . import widgets
def _handle_resize(self, signum=None, frame=None):
    """Tries to catch resize signals sent from the terminal."""
    h, w = array('h', ioctl(self.fd, termios.TIOCGWINSZ, '\x00' * 8))[:2]
    self.term_width = w