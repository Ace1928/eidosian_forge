import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
def wait_for_syn(jid):
    i = 0
    while 1:
        if i > 60:
            error('!!!WAIT FOR ACK TIMEOUT: job:%r fd:%r!!!', jid, self.synq._reader.fileno(), exc_info=1)
        req = _wait_for_syn()
        if req:
            type_, args = req
            if type_ == NACK:
                return False
            assert type_ == ACK
            return True
        i += 1