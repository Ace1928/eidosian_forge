from __future__ import absolute_import
import errno
import os
import time
from . import (LockBase, AlreadyLocked, LockFailed, NotLocked, NotMyLock,
 Break an existing lock.

            Removes the PID file if it already exists, otherwise does
            nothing.

            