import collections.abc
import contextlib
from fnmatch import fnmatch
import gc
import importlib
from io import StringIO
import locale
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any
from typing import Callable
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from weakref import WeakKeyDictionary
from iniconfig import IniConfig
from iniconfig import SectionWrapper
from _pytest import timing
from _pytest._code import Source
from _pytest.capture import _get_multicapture
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import make_numbered_dir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestWarning
class LsofFdLeakChecker:

    def get_open_files(self) -> List[Tuple[str, str]]:
        if sys.version_info >= (3, 11):
            encoding = locale.getencoding()
        else:
            encoding = locale.getpreferredencoding(False)
        out = subprocess.run(('lsof', '-Ffn0', '-p', str(os.getpid())), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, text=True, encoding=encoding).stdout

        def isopen(line: str) -> bool:
            return line.startswith('f') and ('deleted' not in line and 'mem' not in line and ('txt' not in line) and ('cwd' not in line))
        open_files = []
        for line in out.split('\n'):
            if isopen(line):
                fields = line.split('\x00')
                fd = fields[0][1:]
                filename = fields[1][1:]
                if filename in IGNORE_PAM:
                    continue
                if filename.startswith('/'):
                    open_files.append((fd, filename))
        return open_files

    def matching_platform(self) -> bool:
        try:
            subprocess.run(('lsof', '-v'), check=True)
        except (OSError, subprocess.CalledProcessError):
            return False
        else:
            return True

    @hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item: Item) -> Generator[None, object, object]:
        lines1 = self.get_open_files()
        try:
            return (yield)
        finally:
            if hasattr(sys, 'pypy_version_info'):
                gc.collect()
            lines2 = self.get_open_files()
            new_fds = {t[0] for t in lines2} - {t[0] for t in lines1}
            leaked_files = [t for t in lines2 if t[0] in new_fds]
            if leaked_files:
                error = ['***** %s FD leakage detected' % len(leaked_files), *(str(f) for f in leaked_files), '*** Before:', *(str(f) for f in lines1), '*** After:', *(str(f) for f in lines2), '***** %s FD leakage detected' % len(leaked_files), '*** function {}:{}: {} '.format(*item.location), 'See issue #2366']
                item.warn(PytestWarning('\n'.join(error)))