import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
from . import connection
from . import process
from . import reduction
from . import semaphore_tracker
from . import spawn
from . import util
from .compat import spawnv_passfds
def __unpack_fds(child_r, child_w, alive, stfd, *inherited):
    return (child_r, child_w, alive, stfd, inherited)