import functools
import logging
import multiprocessing
import os
import random
import shlex
import signal
import sys
import time
import warnings
import enum
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
class ProcessLimits(object):
    """Resource limits on a process.

    Attributes:

    * address_space: Address space limit in bytes
    * core_file_size: Core file size limit in bytes
    * cpu_time: CPU time limit in seconds
    * data_size: Data size limit in bytes
    * file_size: File size limit in bytes
    * memory_locked: Locked memory limit in bytes
    * number_files: Maximum number of open files
    * number_processes: Maximum number of processes
    * resident_set_size: Maximum Resident Set Size (RSS) in bytes
    * stack_size: Stack size limit in bytes

    This object can be used for the *prlimit* parameter of :func:`execute`.
    """
    _LIMITS = {'address_space': '--as', 'core_file_size': '--core', 'cpu_time': '--cpu', 'data_size': '--data', 'file_size': '--fsize', 'memory_locked': '--memlock', 'number_files': '--nofile', 'number_processes': '--nproc', 'resident_set_size': '--rss', 'stack_size': '--stack'}

    def __init__(self, **kw):
        for limit in self._LIMITS:
            setattr(self, limit, kw.pop(limit, None))
        if kw:
            raise ValueError('invalid limits: %s' % ', '.join(sorted(kw.keys())))

    def prlimit_args(self):
        """Create a list of arguments for the prlimit command line."""
        args = []
        for limit in self._LIMITS:
            val = getattr(self, limit)
            if val is not None:
                args.append('%s=%s' % (self._LIMITS[limit], val))
        return args