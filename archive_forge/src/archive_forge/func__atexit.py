import atexit
import struct
import warnings
from collections import namedtuple
from os import getpid
from threading import Event, Lock, Thread
import zmq
def _atexit(self):
    """atexit callback

        sets _stay_down flag so that gc doesn't try to start up again in other atexit handlers
        """
    self._stay_down = True
    self.stop()