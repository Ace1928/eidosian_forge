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
def assertProcessZombie(self, proc):
    clone = psutil.Process(proc.pid)
    self.assertEqual(proc, clone)
    if not (OPENBSD or NETBSD):
        self.assertEqual(hash(proc), hash(clone))
    self.assertEqual(proc.status(), psutil.STATUS_ZOMBIE)
    assert proc.is_running()
    assert psutil.pid_exists(proc.pid)
    proc.as_dict()
    self.assertIn(proc.pid, psutil.pids())
    self.assertIn(proc.pid, [x.pid for x in psutil.process_iter()])
    psutil._pmap = {}
    self.assertIn(proc.pid, [x.pid for x in psutil.process_iter()])
    ns = process_namespace(proc)
    for fun, name in ns.iter(ns.all, clear_cache=True):
        with self.subTest(proc=proc, name=name):
            try:
                fun()
            except (psutil.ZombieProcess, psutil.AccessDenied) as exc:
                self._check_proc_exc(proc, exc)
    if LINUX:
        with self.assertRaises(psutil.ZombieProcess) as cm:
            proc.cmdline()
        self._check_proc_exc(proc, cm.exception)
        with self.assertRaises(psutil.ZombieProcess) as cm:
            proc.exe()
        self._check_proc_exc(proc, cm.exception)
        with self.assertRaises(psutil.ZombieProcess) as cm:
            proc.memory_maps()
        self._check_proc_exc(proc, cm.exception)
    proc.suspend()
    proc.resume()
    proc.terminate()
    proc.kill()
    assert proc.is_running()
    assert psutil.pid_exists(proc.pid)
    self.assertIn(proc.pid, psutil.pids())
    self.assertIn(proc.pid, [x.pid for x in psutil.process_iter()])
    psutil._pmap = {}
    self.assertIn(proc.pid, [x.pid for x in psutil.process_iter()])