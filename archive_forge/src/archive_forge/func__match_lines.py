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
def _match_lines(self, lines2: Sequence[str], match_func: Callable[[str, str], bool], match_nickname: str, *, consecutive: bool=False) -> None:
    """Underlying implementation of ``fnmatch_lines`` and ``re_match_lines``.

        :param Sequence[str] lines2:
            List of string patterns to match. The actual format depends on
            ``match_func``.
        :param match_func:
            A callable ``match_func(line, pattern)`` where line is the
            captured line from stdout/stderr and pattern is the matching
            pattern.
        :param str match_nickname:
            The nickname for the match function that will be logged to stdout
            when a match occurs.
        :param consecutive:
            Match lines consecutively?
        """
    if not isinstance(lines2, collections.abc.Sequence):
        raise TypeError(f'invalid type for lines2: {type(lines2).__name__}')
    lines2 = self._getlines(lines2)
    lines1 = self.lines[:]
    extralines = []
    __tracebackhide__ = True
    wnick = len(match_nickname) + 1
    started = False
    for line in lines2:
        nomatchprinted = False
        while lines1:
            nextline = lines1.pop(0)
            if line == nextline:
                self._log('exact match:', repr(line))
                started = True
                break
            elif match_func(nextline, line):
                self._log('%s:' % match_nickname, repr(line))
                self._log('{:>{width}}'.format('with:', width=wnick), repr(nextline))
                started = True
                break
            else:
                if consecutive and started:
                    msg = f'no consecutive match: {line!r}'
                    self._log(msg)
                    self._log('{:>{width}}'.format('with:', width=wnick), repr(nextline))
                    self._fail(msg)
                if not nomatchprinted:
                    self._log('{:>{width}}'.format('nomatch:', width=wnick), repr(line))
                    nomatchprinted = True
                self._log('{:>{width}}'.format('and:', width=wnick), repr(nextline))
            extralines.append(nextline)
        else:
            msg = f'remains unmatched: {line!r}'
            self._log(msg)
            self._fail(msg)
    self._log_output = []