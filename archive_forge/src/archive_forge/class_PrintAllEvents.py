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
class PrintAllEvents(ProcessEvent):
    """
    Dummy class used to print events strings representations. For instance this
    class is used from command line to print all received events to stdout.
    """

    def my_init(self, out=None):
        """
        @param out: Where events will be written.
        @type out: Object providing a valid file object interface.
        """
        if out is None:
            out = sys.stdout
        self._out = out

    def process_default(self, event):
        """
        Writes event string representation to file object provided to
        my_init().

        @param event: Event to be processed. Can be of any type of events but
                      IN_Q_OVERFLOW events (see method process_IN_Q_OVERFLOW).
        @type event: Event instance
        """
        self._out.write(str(event))
        self._out.write('\n')
        self._out.flush()