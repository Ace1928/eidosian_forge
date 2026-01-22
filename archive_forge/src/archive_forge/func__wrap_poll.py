import errno
import math
import select
import socket
import sys
import time
from collections import namedtuple
from ansible.module_utils.six.moves.collections_abc import Mapping
def _wrap_poll(self, timeout=None):
    """ Wrapper function for select.poll.poll() so that
            _syscall_wrapper can work with only seconds. """
    if timeout is not None:
        if timeout <= 0:
            timeout = 0
        else:
            timeout = math.ceil(timeout * 1000.0)
    result = self._devpoll.poll(timeout)
    return result