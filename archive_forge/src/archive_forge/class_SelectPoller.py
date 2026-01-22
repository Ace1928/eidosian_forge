import asyncio
from collections import deque
import errno
import fcntl
import gc
import getpass
import glob as glob_module
import inspect
import logging
import os
import platform
import pty
import pwd
import re
import select
import signal
import stat
import struct
import sys
import termios
import textwrap
import threading
import time
import traceback
import tty
import warnings
import weakref
from asyncio import Queue as AQueue
from contextlib import contextmanager
from functools import partial
from importlib import metadata
from io import BytesIO, StringIO, UnsupportedOperation
from io import open as fdopen
from locale import getpreferredencoding
from queue import Empty, Queue
from shlex import quote as shlex_quote
from types import GeneratorType, ModuleType
from typing import Any, Dict, Type, Union
class SelectPoller(object):

    def __init__(self):
        self.rlist = []
        self.wlist = []
        self.xlist = []

    def __nonzero__(self):
        return len(self.rlist) + len(self.wlist) + len(self.xlist) != 0

    def __len__(self):
        return len(self.rlist) + len(self.wlist) + len(self.xlist)

    @staticmethod
    def _register(f, events):
        if f not in events:
            events.append(f)

    @staticmethod
    def _unregister(f, events):
        if f in events:
            events.remove(f)

    def register_read(self, f):
        self._register(f, self.rlist)

    def register_write(self, f):
        self._register(f, self.wlist)

    def register_error(self, f):
        self._register(f, self.xlist)

    def unregister(self, f):
        self._unregister(f, self.rlist)
        self._unregister(f, self.wlist)
        self._unregister(f, self.xlist)

    def poll(self, timeout):
        _in, _out, _err = select.select(self.rlist, self.wlist, self.xlist, timeout)
        results = []
        for f in _in:
            results.append((f, POLLER_EVENT_READ))
        for f in _out:
            results.append((f, POLLER_EVENT_WRITE))
        for f in _err:
            results.append((f, POLLER_EVENT_ERROR))
        return results