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
@dataclasses.dataclass
class FormattedExcinfo:
    """Presenting information about failing Functions and Generators."""
    flow_marker: ClassVar = '>'
    fail_marker: ClassVar = 'E'
    showlocals: bool = False
    style: _TracebackStyle = 'long'
    abspath: bool = True
    tbfilter: Union[bool, Callable[[ExceptionInfo[BaseException]], Traceback]] = True
    funcargs: bool = False
    truncate_locals: bool = True
    chain: bool = True
    astcache: Dict[Union[str, Path], ast.AST] = dataclasses.field(default_factory=dict, init=False, repr=False)

    def _getindent(self, source: 'Source') -> int:
        try:
            s = str(source.getstatement(len(source) - 1))
        except KeyboardInterrupt:
            raise
        except BaseException:
            try:
                s = str(source[-1])
            except KeyboardInterrupt:
                raise
            except BaseException:
                return 0
        return 4 + (len(s) - len(s.lstrip()))

    def _getentrysource(self, entry: TracebackEntry) -> Optional['Source']:
        source = entry.getsource(self.astcache)
        if source is not None:
            source = source.deindent()
        return source

    def repr_args(self, entry: TracebackEntry) -> Optional['ReprFuncArgs']:
        if self.funcargs:
            args = []
            for argname, argvalue in entry.frame.getargs(var=True):
                args.append((argname, saferepr(argvalue)))
            return ReprFuncArgs(args)
        return None

    def get_source(self, source: Optional['Source'], line_index: int=-1, excinfo: Optional[ExceptionInfo[BaseException]]=None, short: bool=False) -> List[str]:
        """Return formatted and marked up source lines."""
        lines = []
        if source is not None and line_index < 0:
            line_index += len(source)
        if source is None or line_index >= len(source.lines) or line_index < 0:
            source = Source('???')
            line_index = 0
        space_prefix = '    '
        if short:
            lines.append(space_prefix + source.lines[line_index].strip())
        else:
            for line in source.lines[:line_index]:
                lines.append(space_prefix + line)
            lines.append(self.flow_marker + '   ' + source.lines[line_index])
            for line in source.lines[line_index + 1:]:
                lines.append(space_prefix + line)
        if excinfo is not None:
            indent = 4 if short else self._getindent(source)
            lines.extend(self.get_exconly(excinfo, indent=indent, markall=True))
        return lines

    def get_exconly(self, excinfo: ExceptionInfo[BaseException], indent: int=4, markall: bool=False) -> List[str]:
        lines = []
        indentstr = ' ' * indent
        exlines = excinfo.exconly(tryshort=True).split('\n')
        failindent = self.fail_marker + indentstr[1:]
        for line in exlines:
            lines.append(failindent + line)
            if not markall:
                failindent = indentstr
        return lines

    def repr_locals(self, locals: Mapping[str, object]) -> Optional['ReprLocals']:
        if self.showlocals:
            lines = []
            keys = [loc for loc in locals if loc[0] != '@']
            keys.sort()
            for name in keys:
                value = locals[name]
                if name == '__builtins__':
                    lines.append('__builtins__ = <builtins>')
                else:
                    if self.truncate_locals:
                        str_repr = saferepr(value)
                    else:
                        str_repr = safeformat(value)
                    lines.append(f'{name:<10} = {str_repr}')
            return ReprLocals(lines)
        return None

    def repr_traceback_entry(self, entry: Optional[TracebackEntry], excinfo: Optional[ExceptionInfo[BaseException]]=None) -> 'ReprEntry':
        lines: List[str] = []
        style = entry._repr_style if entry is not None and entry._repr_style is not None else self.style
        if style in ('short', 'long') and entry is not None:
            source = self._getentrysource(entry)
            if source is None:
                source = Source('???')
                line_index = 0
            else:
                line_index = entry.lineno - entry.getfirstlinesource()
            short = style == 'short'
            reprargs = self.repr_args(entry) if not short else None
            s = self.get_source(source, line_index, excinfo, short=short)
            lines.extend(s)
            if short:
                message = 'in %s' % entry.name
            else:
                message = excinfo and excinfo.typename or ''
            entry_path = entry.path
            path = self._makepath(entry_path)
            reprfileloc = ReprFileLocation(path, entry.lineno + 1, message)
            localsrepr = self.repr_locals(entry.locals)
            return ReprEntry(lines, reprargs, localsrepr, reprfileloc, style)
        elif style == 'value':
            if excinfo:
                lines.extend(str(excinfo.value).split('\n'))
            return ReprEntry(lines, None, None, None, style)
        else:
            if excinfo:
                lines.extend(self.get_exconly(excinfo, indent=4))
            return ReprEntry(lines, None, None, None, style)

    def _makepath(self, path: Union[Path, str]) -> str:
        if not self.abspath and isinstance(path, Path):
            try:
                np = bestrelpath(Path.cwd(), path)
            except OSError:
                return str(path)
            if len(np) < len(str(path)):
                return np
        return str(path)

    def repr_traceback(self, excinfo: ExceptionInfo[BaseException]) -> 'ReprTraceback':
        traceback = excinfo.traceback
        if callable(self.tbfilter):
            traceback = self.tbfilter(excinfo)
        elif self.tbfilter:
            traceback = traceback.filter(excinfo)
        if isinstance(excinfo.value, RecursionError):
            traceback, extraline = self._truncate_recursive_traceback(traceback)
        else:
            extraline = None
        if not traceback:
            if extraline is None:
                extraline = 'All traceback entries are hidden. Pass `--full-trace` to see hidden and internal frames.'
            entries = [self.repr_traceback_entry(None, excinfo)]
            return ReprTraceback(entries, extraline, style=self.style)
        last = traceback[-1]
        if self.style == 'value':
            entries = [self.repr_traceback_entry(last, excinfo)]
            return ReprTraceback(entries, None, style=self.style)
        entries = [self.repr_traceback_entry(entry, excinfo if last == entry else None) for entry in traceback]
        return ReprTraceback(entries, extraline, style=self.style)

    def _truncate_recursive_traceback(self, traceback: Traceback) -> Tuple[Traceback, Optional[str]]:
        """Truncate the given recursive traceback trying to find the starting
        point of the recursion.

        The detection is done by going through each traceback entry and
        finding the point in which the locals of the frame are equal to the
        locals of a previous frame (see ``recursionindex()``).

        Handle the situation where the recursion process might raise an
        exception (for example comparing numpy arrays using equality raises a
        TypeError), in which case we do our best to warn the user of the
        error and show a limited traceback.
        """
        try:
            recursionindex = traceback.recursionindex()
        except Exception as e:
            max_frames = 10
            extraline: Optional[str] = f'!!! Recursion error detected, but an error occurred locating the origin of recursion.\n  The following exception happened when comparing locals in the stack frame:\n    {type(e).__name__}: {e!s}\n  Displaying first and last {max_frames} stack frames out of {len(traceback)}.'
            traceback = traceback[:max_frames] + traceback[-max_frames:]
        else:
            if recursionindex is not None:
                extraline = '!!! Recursion detected (same locals & position)'
                traceback = traceback[:recursionindex + 1]
            else:
                extraline = None
        return (traceback, extraline)

    def repr_excinfo(self, excinfo: ExceptionInfo[BaseException]) -> 'ExceptionChainRepr':
        repr_chain: List[Tuple[ReprTraceback, Optional[ReprFileLocation], Optional[str]]] = []
        e: Optional[BaseException] = excinfo.value
        excinfo_: Optional[ExceptionInfo[BaseException]] = excinfo
        descr = None
        seen: Set[int] = set()
        while e is not None and id(e) not in seen:
            seen.add(id(e))
            if excinfo_:
                if isinstance(e, BaseExceptionGroup):
                    reprtraceback: Union[ReprTracebackNative, ReprTraceback] = ReprTracebackNative(traceback.format_exception(type(excinfo_.value), excinfo_.value, excinfo_.traceback[0]._rawentry))
                else:
                    reprtraceback = self.repr_traceback(excinfo_)
                reprcrash = excinfo_._getreprcrash()
            else:
                reprtraceback = ReprTracebackNative(traceback.format_exception(type(e), e, None))
                reprcrash = None
            repr_chain += [(reprtraceback, reprcrash, descr)]
            if e.__cause__ is not None and self.chain:
                e = e.__cause__
                excinfo_ = ExceptionInfo.from_exception(e) if e.__traceback__ else None
                descr = 'The above exception was the direct cause of the following exception:'
            elif e.__context__ is not None and (not e.__suppress_context__) and self.chain:
                e = e.__context__
                excinfo_ = ExceptionInfo.from_exception(e) if e.__traceback__ else None
                descr = 'During handling of the above exception, another exception occurred:'
            else:
                e = None
        repr_chain.reverse()
        return ExceptionChainRepr(repr_chain)