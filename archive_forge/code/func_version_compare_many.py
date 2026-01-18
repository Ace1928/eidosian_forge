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
def version_compare_many(vstr1: str, conditions: T.Union[str, T.Iterable[str]]) -> T.Tuple[bool, T.List[str], T.List[str]]:
    if isinstance(conditions, str):
        conditions = [conditions]
    found: T.List[str] = []
    not_found: T.List[str] = []
    for req in conditions:
        if not version_compare(vstr1, req):
            not_found.append(req)
        else:
            found.append(req)
    return (not not_found, not_found, found)