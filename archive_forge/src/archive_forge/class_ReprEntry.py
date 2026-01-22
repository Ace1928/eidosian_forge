import ast
import dataclasses
import inspect
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
import os
from pathlib import Path
import re
import sys
import traceback
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import final
from typing import Generic
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import SupportsIndex
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import pluggy
import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import get_real_func
from _pytest.deprecated import check_ispytest
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
@dataclasses.dataclass(eq=False)
class ReprEntry(TerminalRepr):
    lines: Sequence[str]
    reprfuncargs: Optional['ReprFuncArgs']
    reprlocals: Optional['ReprLocals']
    reprfileloc: Optional['ReprFileLocation']
    style: _TracebackStyle

    def _write_entry_lines(self, tw: TerminalWriter) -> None:
        """Write the source code portions of a list of traceback entries with syntax highlighting.

        Usually entries are lines like these:

            "     x = 1"
            ">    assert x == 2"
            "E    assert 1 == 2"

        This function takes care of rendering the "source" portions of it (the lines without
        the "E" prefix) using syntax highlighting, taking care to not highlighting the ">"
        character, as doing so might break line continuations.
        """
        if not self.lines:
            return
        fail_marker = f'{FormattedExcinfo.fail_marker}   '
        indent_size = len(fail_marker)
        indents: List[str] = []
        source_lines: List[str] = []
        failure_lines: List[str] = []
        for index, line in enumerate(self.lines):
            is_failure_line = line.startswith(fail_marker)
            if is_failure_line:
                failure_lines.extend(self.lines[index:])
                break
            elif self.style == 'value':
                source_lines.append(line)
            else:
                indents.append(line[:indent_size])
                source_lines.append(line[indent_size:])
        tw._write_source(source_lines, indents)
        for line in failure_lines:
            tw.line(line, bold=True, red=True)

    def toterminal(self, tw: TerminalWriter) -> None:
        if self.style == 'short':
            if self.reprfileloc:
                self.reprfileloc.toterminal(tw)
            self._write_entry_lines(tw)
            if self.reprlocals:
                self.reprlocals.toterminal(tw, indent=' ' * 8)
            return
        if self.reprfuncargs:
            self.reprfuncargs.toterminal(tw)
        self._write_entry_lines(tw)
        if self.reprlocals:
            tw.line('')
            self.reprlocals.toterminal(tw)
        if self.reprfileloc:
            if self.lines:
                tw.line('')
            self.reprfileloc.toterminal(tw)

    def __str__(self) -> str:
        return '{}\n{}\n{}'.format('\n'.join(self.lines), self.reprlocals, self.reprfileloc)