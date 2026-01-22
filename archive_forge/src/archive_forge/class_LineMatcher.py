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
class LineMatcher:
    """Flexible matching of text.

    This is a convenience class to test large texts like the output of
    commands.

    The constructor takes a list of lines without their trailing newlines, i.e.
    ``text.splitlines()``.
    """

    def __init__(self, lines: List[str]) -> None:
        self.lines = lines
        self._log_output: List[str] = []

    def __str__(self) -> str:
        """Return the entire original text.

        .. versionadded:: 6.2
            You can use :meth:`str` in older versions.
        """
        return '\n'.join(self.lines)

    def _getlines(self, lines2: Union[str, Sequence[str], Source]) -> Sequence[str]:
        if isinstance(lines2, str):
            lines2 = Source(lines2)
        if isinstance(lines2, Source):
            lines2 = lines2.strip().lines
        return lines2

    def fnmatch_lines_random(self, lines2: Sequence[str]) -> None:
        """Check lines exist in the output in any order (using :func:`python:fnmatch.fnmatch`)."""
        __tracebackhide__ = True
        self._match_lines_random(lines2, fnmatch)

    def re_match_lines_random(self, lines2: Sequence[str]) -> None:
        """Check lines exist in the output in any order (using :func:`python:re.match`)."""
        __tracebackhide__ = True
        self._match_lines_random(lines2, lambda name, pat: bool(re.match(pat, name)))

    def _match_lines_random(self, lines2: Sequence[str], match_func: Callable[[str, str], bool]) -> None:
        __tracebackhide__ = True
        lines2 = self._getlines(lines2)
        for line in lines2:
            for x in self.lines:
                if line == x or match_func(x, line):
                    self._log('matched: ', repr(line))
                    break
            else:
                msg = 'line %r not found in output' % line
                self._log(msg)
                self._fail(msg)

    def get_lines_after(self, fnline: str) -> Sequence[str]:
        """Return all lines following the given line in the text.

        The given line can contain glob wildcards.
        """
        for i, line in enumerate(self.lines):
            if fnline == line or fnmatch(line, fnline):
                return self.lines[i + 1:]
        raise ValueError('line %r not found in output' % fnline)

    def _log(self, *args) -> None:
        self._log_output.append(' '.join((str(x) for x in args)))

    @property
    def _log_text(self) -> str:
        return '\n'.join(self._log_output)

    def fnmatch_lines(self, lines2: Sequence[str], *, consecutive: bool=False) -> None:
        """Check lines exist in the output (using :func:`python:fnmatch.fnmatch`).

        The argument is a list of lines which have to match and can use glob
        wildcards.  If they do not match a pytest.fail() is called.  The
        matches and non-matches are also shown as part of the error message.

        :param lines2: String patterns to match.
        :param consecutive: Match lines consecutively?
        """
        __tracebackhide__ = True
        self._match_lines(lines2, fnmatch, 'fnmatch', consecutive=consecutive)

    def re_match_lines(self, lines2: Sequence[str], *, consecutive: bool=False) -> None:
        """Check lines exist in the output (using :func:`python:re.match`).

        The argument is a list of lines which have to match using ``re.match``.
        If they do not match a pytest.fail() is called.

        The matches and non-matches are also shown as part of the error message.

        :param lines2: string patterns to match.
        :param consecutive: match lines consecutively?
        """
        __tracebackhide__ = True
        self._match_lines(lines2, lambda name, pat: bool(re.match(pat, name)), 're.match', consecutive=consecutive)

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

    def no_fnmatch_line(self, pat: str) -> None:
        """Ensure captured lines do not match the given pattern, using ``fnmatch.fnmatch``.

        :param str pat: The pattern to match lines.
        """
        __tracebackhide__ = True
        self._no_match_line(pat, fnmatch, 'fnmatch')

    def no_re_match_line(self, pat: str) -> None:
        """Ensure captured lines do not match the given pattern, using ``re.match``.

        :param str pat: The regular expression to match lines.
        """
        __tracebackhide__ = True
        self._no_match_line(pat, lambda name, pat: bool(re.match(pat, name)), 're.match')

    def _no_match_line(self, pat: str, match_func: Callable[[str, str], bool], match_nickname: str) -> None:
        """Ensure captured lines does not have a the given pattern, using ``fnmatch.fnmatch``.

        :param str pat: The pattern to match lines.
        """
        __tracebackhide__ = True
        nomatch_printed = False
        wnick = len(match_nickname) + 1
        for line in self.lines:
            if match_func(line, pat):
                msg = f'{match_nickname}: {pat!r}'
                self._log(msg)
                self._log('{:>{width}}'.format('with:', width=wnick), repr(line))
                self._fail(msg)
            else:
                if not nomatch_printed:
                    self._log('{:>{width}}'.format('nomatch:', width=wnick), repr(pat))
                    nomatch_printed = True
                self._log('{:>{width}}'.format('and:', width=wnick), repr(line))
        self._log_output = []

    def _fail(self, msg: str) -> None:
        __tracebackhide__ = True
        log_text = self._log_text
        self._log_output = []
        fail(log_text)

    def str(self) -> str:
        """Return the entire original text."""
        return str(self)