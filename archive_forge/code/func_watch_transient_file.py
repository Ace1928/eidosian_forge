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
def watch_transient_file(self, filename, mask, proc_class):
    """
        Watch a transient file, which will be created and deleted frequently
        over time (e.g. pid file).

        @attention: Currently under the call to this function it is not
        possible to correctly watch the events triggered into the same
        base directory than the directory where is located this watched
        transient file. For instance it would be wrong to make these
        two successive calls: wm.watch_transient_file('/var/run/foo.pid', ...)
        and wm.add_watch('/var/run/', ...)

        @param filename: Filename.
        @type filename: string
        @param mask: Bitmask of events, should contain IN_CREATE and IN_DELETE.
        @type mask: int
        @param proc_class: ProcessEvent (or of one of its subclass), beware of
                           accepting a ProcessEvent's instance as argument into
                           __init__, see transient_file.py example for more
                           details.
        @type proc_class: ProcessEvent's instance or of one of its subclasses.
        @return: Same as add_watch().
        @rtype: Same as add_watch().
        """
    dirname = os.path.dirname(filename)
    if dirname == '':
        return {}
    basename = os.path.basename(filename)
    mask |= IN_CREATE | IN_DELETE

    def cmp_name(event):
        if getattr(event, 'name') is None:
            return False
        return basename == event.name
    return self.add_watch(dirname, mask, proc_fun=proc_class(ChainIfTrue(func=cmp_name)), rec=False, auto_add=False, do_glob=False, exclude_filter=lambda path: False)