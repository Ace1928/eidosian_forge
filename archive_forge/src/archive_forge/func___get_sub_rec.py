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
def __get_sub_rec(self, lpath):
    """
        Get every wd from self._wmd if its path is under the path of
        one (at least) of those in lpath. Doesn't follow symlinks.

        @param lpath: list of watch descriptor
        @type lpath: list of int
        @return: list of watch descriptor
        @rtype: list of int
        """
    for d in lpath:
        root = self.get_path(d)
        if root is not None:
            yield d
        else:
            continue
        if not os.path.isdir(root):
            continue
        root = os.path.normpath(root)
        lend = len(root)
        for iwd in self._wmd.items():
            cur = iwd[1].path
            pref = os.path.commonprefix([root, cur])
            if root == os.sep or (len(pref) == lend and len(cur) > lend and (cur[lend] == os.sep)):
                yield iwd[1].wd