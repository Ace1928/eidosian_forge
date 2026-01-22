from __future__ import division
import base64
import collections
import errno
import functools
import glob
import os
import re
import socket
import struct
import sys
import warnings
from collections import defaultdict
from collections import namedtuple
from . import _common
from . import _psposix
from . import _psutil_linux as cext
from . import _psutil_posix as cext_posix
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import bcat
from ._common import cat
from ._common import debug
from ._common import decode
from ._common import get_procfs_path
from ._common import isfile_strict
from ._common import memoize
from ._common import memoize_when_activated
from ._common import open_binary
from ._common import open_text
from ._common import parse_environ_block
from ._common import path_exists_strict
from ._common import supports_ipv6
from ._common import usage_percent
from ._compat import PY3
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import b
from ._compat import basestring
class RootFsDeviceFinder:
    """disk_partitions() may return partitions with device == "/dev/root"
    or "rootfs". This container class uses different strategies to try to
    obtain the real device path. Resources:
    https://bootlin.com/blog/find-root-device/
    https://www.systutorials.com/how-to-find-the-disk-where-root-is-on-in-bash-on-linux/.
    """
    __slots__ = ['major', 'minor']

    def __init__(self):
        dev = os.stat('/').st_dev
        self.major = os.major(dev)
        self.minor = os.minor(dev)

    def ask_proc_partitions(self):
        with open_text('%s/partitions' % get_procfs_path()) as f:
            for line in f.readlines()[2:]:
                fields = line.split()
                if len(fields) < 4:
                    continue
                major = int(fields[0]) if fields[0].isdigit() else None
                minor = int(fields[1]) if fields[1].isdigit() else None
                name = fields[3]
                if major == self.major and minor == self.minor:
                    if name:
                        return '/dev/%s' % name

    def ask_sys_dev_block(self):
        path = '/sys/dev/block/%s:%s/uevent' % (self.major, self.minor)
        with open_text(path) as f:
            for line in f:
                if line.startswith('DEVNAME='):
                    name = line.strip().rpartition('DEVNAME=')[2]
                    if name:
                        return '/dev/%s' % name

    def ask_sys_class_block(self):
        needle = '%s:%s' % (self.major, self.minor)
        files = glob.iglob('/sys/class/block/*/dev')
        for file in files:
            try:
                f = open_text(file)
            except FileNotFoundError:
                continue
            else:
                with f:
                    data = f.read().strip()
                    if data == needle:
                        name = os.path.basename(os.path.dirname(file))
                        return '/dev/%s' % name

    def find(self):
        path = None
        if path is None:
            try:
                path = self.ask_proc_partitions()
            except (IOError, OSError) as err:
                debug(err)
        if path is None:
            try:
                path = self.ask_sys_dev_block()
            except (IOError, OSError) as err:
                debug(err)
        if path is None:
            try:
                path = self.ask_sys_class_block()
            except (IOError, OSError) as err:
                debug(err)
        if path is not None and os.path.exists(path):
            return path