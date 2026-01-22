from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
def _execChild(self, path, uid, gid, executable, args, environment):
    """
        The exec() which is done in the forked child.
        """
    if path:
        os.chdir(path)
    if uid is not None or gid is not None:
        if uid is None:
            uid = os.geteuid()
        if gid is None:
            gid = os.getegid()
        os.setuid(0)
        os.setgid(0)
        switchUID(uid, gid)
    os.execvpe(executable, args, environment)