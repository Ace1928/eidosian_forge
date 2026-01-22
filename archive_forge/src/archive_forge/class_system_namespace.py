from __future__ import print_function
import atexit
import contextlib
import ctypes
import errno
import functools
import gc
import inspect
import os
import platform
import random
import re
import select
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_STREAM
import psutil
from psutil import AIX
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import bytes2human
from psutil._common import debug
from psutil._common import memoize
from psutil._common import print_color
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil._compat import FileExistsError
from psutil._compat import FileNotFoundError
from psutil._compat import range
from psutil._compat import super
from psutil._compat import u
from psutil._compat import unicode
from psutil._compat import which
class system_namespace:
    """A container that lists all the module-level, system-related APIs.
    Utilities such as cpu_percent() are excluded. Usage:

    >>> ns = system_namespace
    >>> for fun, name in ns.iter(ns.getters):
    ...    fun()
    """
    getters = [('boot_time', (), {}), ('cpu_count', (), {'logical': False}), ('cpu_count', (), {'logical': True}), ('cpu_stats', (), {}), ('cpu_times', (), {'percpu': False}), ('cpu_times', (), {'percpu': True}), ('disk_io_counters', (), {'perdisk': True}), ('disk_partitions', (), {'all': True}), ('disk_usage', (os.getcwd(),), {}), ('net_connections', (), {'kind': 'all'}), ('net_if_addrs', (), {}), ('net_if_stats', (), {}), ('net_io_counters', (), {'pernic': True}), ('pid_exists', (os.getpid(),), {}), ('pids', (), {}), ('swap_memory', (), {}), ('users', (), {}), ('virtual_memory', (), {})]
    if HAS_CPU_FREQ:
        getters += [('cpu_freq', (), {'percpu': True})]
    if HAS_GETLOADAVG:
        getters += [('getloadavg', (), {})]
    if HAS_SENSORS_TEMPERATURES:
        getters += [('sensors_temperatures', (), {})]
    if HAS_SENSORS_FANS:
        getters += [('sensors_fans', (), {})]
    if HAS_SENSORS_BATTERY:
        getters += [('sensors_battery', (), {})]
    if WINDOWS:
        getters += [('win_service_iter', (), {})]
        getters += [('win_service_get', ('alg',), {})]
    ignored = [('process_iter', (), {}), ('wait_procs', ([psutil.Process()],), {}), ('cpu_percent', (), {}), ('cpu_times_percent', (), {})]
    all = getters

    @staticmethod
    def iter(ls):
        """Given a list of tuples yields a set of (fun, fun_name) tuples
        in random order.
        """
        ls = list(ls)
        random.shuffle(ls)
        for fun_name, args, kwds in ls:
            fun = getattr(psutil, fun_name)
            fun = functools.partial(fun, *args, **kwds)
            yield (fun, fun_name)
    test_class_coverage = process_namespace.test_class_coverage