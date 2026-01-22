import contextlib
import errno
import functools
import os
import signal
import sys
import time
from collections import namedtuple
from . import _common
from ._common import ENCODING
from ._common import ENCODING_ERRS
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import TimeoutExpired
from ._common import conn_tmap
from ._common import conn_to_ntuple
from ._common import debug
from ._common import isfile_strict
from ._common import memoize
from ._common import memoize_when_activated
from ._common import parse_environ_block
from ._common import usage_percent
from ._compat import PY3
from ._compat import long
from ._compat import lru_cache
from ._compat import range
from ._compat import unicode
from ._psutil_windows import ABOVE_NORMAL_PRIORITY_CLASS
from ._psutil_windows import BELOW_NORMAL_PRIORITY_CLASS
from ._psutil_windows import HIGH_PRIORITY_CLASS
from ._psutil_windows import IDLE_PRIORITY_CLASS
from ._psutil_windows import NORMAL_PRIORITY_CLASS
from ._psutil_windows import REALTIME_PRIORITY_CLASS
def _get_raw_meminfo(self):
    try:
        return cext.proc_memory_info(self.pid)
    except OSError as err:
        if is_permission_err(err):
            info = self._proc_info()
            return (info[pinfo_map['num_page_faults']], info[pinfo_map['peak_wset']], info[pinfo_map['wset']], info[pinfo_map['peak_paged_pool']], info[pinfo_map['paged_pool']], info[pinfo_map['peak_non_paged_pool']], info[pinfo_map['non_paged_pool']], info[pinfo_map['pagefile']], info[pinfo_map['peak_pagefile']], info[pinfo_map['mem_private']])
        raise