import errno
import os
import select
import socket
import sys
import ovs.timeval
import ovs.vlog
def timer_wait(self, msec):
    """Causes the following call to self.block() to block for no more than
        'msec' milliseconds.  If 'msec' is nonpositive, the following call to
        self.block() will not block at all.

        The timer registration is one-shot: only the following call to
        self.block() is affected.  The timer will need to be re-registered
        after self.block() is called if it is to persist."""
    if msec <= 0:
        self.immediate_wake()
    else:
        self.__timer_wait(msec)