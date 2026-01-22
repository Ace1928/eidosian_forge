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
@wrap_exceptions
@memoize_when_activated
def _parse_stat_file(self):
    """Parse /proc/{pid}/stat file and return a dict with various
        process info.
        Using "man proc" as a reference: where "man proc" refers to
        position N always subtract 3 (e.g ppid position 4 in
        'man proc' == position 1 in here).
        The return value is cached in case oneshot() ctx manager is
        in use.
        """
    data = bcat('%s/%s/stat' % (self._procfs_path, self.pid))
    rpar = data.rfind(b')')
    name = data[data.find(b'(') + 1:rpar]
    fields = data[rpar + 2:].split()
    ret = {}
    ret['name'] = name
    ret['status'] = fields[0]
    ret['ppid'] = fields[1]
    ret['ttynr'] = fields[4]
    ret['utime'] = fields[11]
    ret['stime'] = fields[12]
    ret['children_utime'] = fields[13]
    ret['children_stime'] = fields[14]
    ret['create_time'] = fields[19]
    ret['cpu_num'] = fields[36]
    ret['blkio_ticks'] = fields[39]
    return ret