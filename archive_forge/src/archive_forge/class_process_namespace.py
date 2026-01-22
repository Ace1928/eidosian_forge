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
class process_namespace:
    """A container that lists all Process class method names + some
    reasonable parameters to be called with. Utility methods (parent(),
    children(), ...) are excluded.

    >>> ns = process_namespace(psutil.Process())
    >>> for fun, name in ns.iter(ns.getters):
    ...    fun()
    """
    utils = [('cpu_percent', (), {}), ('memory_percent', (), {})]
    ignored = [('as_dict', (), {}), ('children', (), {'recursive': True}), ('is_running', (), {}), ('memory_info_ex', (), {}), ('oneshot', (), {}), ('parent', (), {}), ('parents', (), {}), ('pid', (), {}), ('wait', (0,), {})]
    getters = [('cmdline', (), {}), ('connections', (), {'kind': 'all'}), ('cpu_times', (), {}), ('create_time', (), {}), ('cwd', (), {}), ('exe', (), {}), ('memory_full_info', (), {}), ('memory_info', (), {}), ('name', (), {}), ('nice', (), {}), ('num_ctx_switches', (), {}), ('num_threads', (), {}), ('open_files', (), {}), ('ppid', (), {}), ('status', (), {}), ('threads', (), {}), ('username', (), {})]
    if POSIX:
        getters += [('uids', (), {})]
        getters += [('gids', (), {})]
        getters += [('terminal', (), {})]
        getters += [('num_fds', (), {})]
    if HAS_PROC_IO_COUNTERS:
        getters += [('io_counters', (), {})]
    if HAS_IONICE:
        getters += [('ionice', (), {})]
    if HAS_RLIMIT:
        getters += [('rlimit', (psutil.RLIMIT_NOFILE,), {})]
    if HAS_CPU_AFFINITY:
        getters += [('cpu_affinity', (), {})]
    if HAS_PROC_CPU_NUM:
        getters += [('cpu_num', (), {})]
    if HAS_ENVIRON:
        getters += [('environ', (), {})]
    if WINDOWS:
        getters += [('num_handles', (), {})]
    if HAS_MEMORY_MAPS:
        getters += [('memory_maps', (), {'grouped': False})]
    setters = []
    if POSIX:
        setters += [('nice', (0,), {})]
    else:
        setters += [('nice', (psutil.NORMAL_PRIORITY_CLASS,), {})]
    if HAS_RLIMIT:
        setters += [('rlimit', (psutil.RLIMIT_NOFILE, (1024, 4096)), {})]
    if HAS_IONICE:
        if LINUX:
            setters += [('ionice', (psutil.IOPRIO_CLASS_NONE, 0), {})]
        else:
            setters += [('ionice', (psutil.IOPRIO_NORMAL,), {})]
    if HAS_CPU_AFFINITY:
        setters += [('cpu_affinity', ([_get_eligible_cpu()],), {})]
    killers = [('send_signal', (signal.SIGTERM,), {}), ('suspend', (), {}), ('resume', (), {}), ('terminate', (), {}), ('kill', (), {})]
    if WINDOWS:
        killers += [('send_signal', (signal.CTRL_C_EVENT,), {})]
        killers += [('send_signal', (signal.CTRL_BREAK_EVENT,), {})]
    all = utils + getters + setters + killers

    def __init__(self, proc):
        self._proc = proc

    def iter(self, ls, clear_cache=True):
        """Given a list of tuples yields a set of (fun, fun_name) tuples
        in random order.
        """
        ls = list(ls)
        random.shuffle(ls)
        for fun_name, args, kwds in ls:
            if clear_cache:
                self.clear_cache()
            fun = getattr(self._proc, fun_name)
            fun = functools.partial(fun, *args, **kwds)
            yield (fun, fun_name)

    def clear_cache(self):
        """Clear the cache of a Process instance."""
        self._proc._init(self._proc.pid, _ignore_nsp=True)

    @classmethod
    def test_class_coverage(cls, test_class, ls):
        """Given a TestCase instance and a list of tuples checks that
        the class defines the required test method names.
        """
        for fun_name, _, _ in ls:
            meth_name = 'test_' + fun_name
            if not hasattr(test_class, meth_name):
                msg = "%r class should define a '%s' method" % (test_class.__class__.__name__, meth_name)
                raise AttributeError(msg)

    @classmethod
    def test(cls):
        this = set([x[0] for x in cls.all])
        ignored = set([x[0] for x in cls.ignored])
        klass = set([x for x in dir(psutil.Process) if x[0] != '_'])
        leftout = (this | ignored) ^ klass
        if leftout:
            raise ValueError('uncovered Process class names: %r' % leftout)