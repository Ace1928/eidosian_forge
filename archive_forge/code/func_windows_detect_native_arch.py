from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
def windows_detect_native_arch() -> str:
    """
    The architecture of Windows itself: x86, amd64 or arm64
    """
    if sys.platform != 'win32':
        return ''
    try:
        import ctypes
        process_arch = ctypes.c_ushort()
        native_arch = ctypes.c_ushort()
        kernel32 = ctypes.windll.kernel32
        process = ctypes.c_void_p(kernel32.GetCurrentProcess())
        if kernel32.IsWow64Process2(process, ctypes.byref(process_arch), ctypes.byref(native_arch)):
            if native_arch.value == 34404:
                return 'amd64'
            elif native_arch.value == 332:
                return 'x86'
            elif native_arch.value == 43620:
                return 'arm64'
            elif native_arch.value == 452:
                return 'arm'
    except (OSError, AttributeError):
        pass
    arch = os.environ.get('PROCESSOR_ARCHITEW6432', '').lower()
    if not arch:
        try:
            arch = os.environ['PROCESSOR_ARCHITECTURE'].lower()
        except KeyError:
            raise EnvironmentException('Unable to detect native OS architecture')
    return arch