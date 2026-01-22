import ctypes
import os
import signal
import struct
import threading
from pyglet.libs.x11 import xlib
from pyglet.util import asbytes
def _sighup(signum, frame):
    parent_wait_lock.release()