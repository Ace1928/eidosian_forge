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
class ErrorReturnCodeMeta(type):
    """a metaclass which provides the ability for an ErrorReturnCode (or
    derived) instance, imported from one sh module, to be considered the
    subclass of ErrorReturnCode from another module.  this is mostly necessary
    in the tests, where we do assertRaises, but the ErrorReturnCode that the
    program we're testing throws may not be the same class that we pass to
    assertRaises
    """

    def __subclasscheck__(self, o):
        other_bases = set([b.__name__ for b in o.__bases__])
        return self.__name__ in other_bases or o.__name__ == self.__name__