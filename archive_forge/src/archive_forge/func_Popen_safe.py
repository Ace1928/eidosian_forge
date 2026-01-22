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
def Popen_safe(args: T.List[str], write: T.Optional[str]=None, stdin: T.Union[None, T.TextIO, T.BinaryIO, int]=subprocess.DEVNULL, stdout: T.Union[None, T.TextIO, T.BinaryIO, int]=subprocess.PIPE, stderr: T.Union[None, T.TextIO, T.BinaryIO, int]=subprocess.PIPE, **kwargs: T.Any) -> T.Tuple['subprocess.Popen[str]', str, str]:
    import locale
    encoding = locale.getpreferredencoding()
    if write is not None:
        stdin = subprocess.PIPE
    try:
        if not sys.stdout.encoding or encoding.upper() != 'UTF-8':
            p, o, e = Popen_safe_legacy(args, write=write, stdin=stdin, stdout=stdout, stderr=stderr, **kwargs)
        else:
            p = subprocess.Popen(args, universal_newlines=True, encoding=encoding, close_fds=False, stdin=stdin, stdout=stdout, stderr=stderr, **kwargs)
            o, e = p.communicate(write)
    except OSError as oserr:
        if oserr.errno == errno.ENOEXEC:
            raise MesonException(f'Failed running {args[0]!r}, binary or interpreter not executable.\nPossibly wrong architecture or the executable bit is not set.')
        raise
    mlog.setup_console()
    return (p, o, e)