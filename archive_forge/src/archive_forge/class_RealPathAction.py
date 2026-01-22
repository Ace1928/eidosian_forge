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
class RealPathAction(argparse.Action):

    def __init__(self, option_strings: T.List[str], dest: str, default: str='.', **kwargs: T.Any):
        default = os.path.abspath(os.path.realpath(default))
        super().__init__(option_strings, dest, nargs=None, default=default, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: T.Union[str, T.Sequence[T.Any], None], option_string: T.Optional[str]=None) -> None:
        assert isinstance(values, str)
        setattr(namespace, self.dest, os.path.abspath(os.path.realpath(values)))