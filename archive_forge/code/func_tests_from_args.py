from __future__ import annotations
from pathlib import Path
from collections import deque
from contextlib import suppress
from copy import deepcopy
from fnmatch import fnmatch
import argparse
import asyncio
import datetime
import enum
import json
import multiprocessing
import os
import pickle
import platform
import random
import re
import signal
import subprocess
import shlex
import sys
import textwrap
import time
import typing as T
import unicodedata
import xml.etree.ElementTree as et
from . import build
from . import environment
from . import mlog
from .coredata import MesonVersionMismatchException, major_versions_differ
from .coredata import version as coredata_version
from .mesonlib import (MesonException, OptionKey, OrderedSet, RealPathAction,
from .mintro import get_infodir, load_info_file
from .programs import ExternalProgram
from .backend.backends import TestProtocol, TestSerialisation
def tests_from_args(self, tests: T.List[TestSerialisation]) -> T.Generator[TestSerialisation, None, None]:
    """
        Allow specifying test names like "meson test foo1 foo2", where test('foo1', ...)

        Also support specifying the subproject to run tests from like
        "meson test subproj:" (all tests inside subproj) or "meson test subproj:foo1"
        to run foo1 inside subproj. Coincidentally also "meson test :foo1" to
        run all tests with that name across all subprojects, which is
        identical to "meson test foo1"
        """
    patterns: T.Dict[T.Tuple[str, str], bool] = {}
    for arg in self.options.args:
        if ':' in arg:
            subproj, name = arg.split(':', maxsplit=1)
            if name == '':
                name = '*'
            if subproj == '':
                subproj = '*'
        else:
            subproj, name = ('*', arg)
        patterns[subproj, name] = False
    for t in tests:
        for subproj, name in list(patterns):
            if fnmatch(t.project_name, subproj) and fnmatch(t.name, name):
                patterns[subproj, name] = True
                yield t
                break
    for (subproj, name), was_used in patterns.items():
        if not was_used:
            arg = f'{subproj}:{name}'
            for t in tests:
                if fnmatch(t.project_name, subproj) and fnmatch(t.name, name):
                    mlog.warning(f'{arg} test name is redundant and was not used')
                    break
            else:
                raise MesonException(f'{arg} test name does not match any test')