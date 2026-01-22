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
class SelfWrapper(ModuleType):

    def __init__(self, self_module, baked_args=None):
        super(SelfWrapper, self).__init__(name=getattr(self_module, '__name__', None), doc=getattr(self_module, '__doc__', None))
        for attr in ['__builtins__', '__file__', '__package__']:
            setattr(self, attr, getattr(self_module, attr, None))
        self.__path__ = []
        self.__self_module = self_module
        command_cls = Command
        cls_attrs = command_cls.__dict__.copy()
        cls_attrs.pop('__dict__', None)
        if baked_args:
            call_args, _ = command_cls._extract_call_args(baked_args)
            cls_attrs['_call_args'] = cls_attrs['_call_args'].copy()
            cls_attrs['_call_args'].update(call_args)
        globs = globals().copy()
        globs[command_cls.__name__] = type(command_cls.__name__, command_cls.__bases__, cls_attrs)
        self.__env = Environment(globs, baked_args=baked_args)

    def __getattr__(self, name):
        return self.__env[name]

    def bake(self, **kwargs):
        baked_args = self.__env.baked_args.copy()
        baked_args.update(kwargs)
        new_sh = self.__class__(self.__self_module, baked_args)
        return new_sh