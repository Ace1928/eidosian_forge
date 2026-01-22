from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
class DebugTraceback:
    __slots__ = ('_te', '_cache_all_tracebacks', '_cache_all_frames')

    def __init__(self, exc: BaseException, te: traceback.TracebackException | None=None, *, skip: int=0, hide: bool=True) -> None:
        self._te = _process_traceback(exc, te, skip=skip, hide=hide)

    def __str__(self) -> str:
        return f'<{type(self).__name__} {self._te}>'

    @cached_property
    def all_tracebacks(self) -> list[tuple[str | None, traceback.TracebackException]]:
        out = []
        current = self._te
        while current is not None:
            if current.__cause__ is not None:
                chained_msg = 'The above exception was the direct cause of the following exception'
                chained_exc = current.__cause__
            elif current.__context__ is not None and (not current.__suppress_context__):
                chained_msg = 'During handling of the above exception, another exception occurred'
                chained_exc = current.__context__
            else:
                chained_msg = None
                chained_exc = None
            out.append((chained_msg, current))
            current = chained_exc
        return out

    @cached_property
    def all_frames(self) -> list[DebugFrameSummary]:
        return [f for _, te in self.all_tracebacks for f in te.stack]

    def render_traceback_text(self) -> str:
        return ''.join(self._te.format())

    def render_traceback_html(self, include_title: bool=True) -> str:
        library_frames = [f.is_library for f in self.all_frames]
        mark_library = 0 < sum(library_frames) < len(library_frames)
        rows = []
        if not library_frames:
            classes = 'traceback noframe-traceback'
        else:
            classes = 'traceback'
            for msg, current in reversed(self.all_tracebacks):
                row_parts = []
                if msg is not None:
                    row_parts.append(f'<li><div class="exc-divider">{msg}:</div>')
                for frame in current.stack:
                    frame = t.cast(DebugFrameSummary, frame)
                    info = f' title="{escape(frame.info)}"' if frame.info else ''
                    row_parts.append(f'<li{info}>{frame.render_html(mark_library)}')
                rows.append('\n'.join(row_parts))
        is_syntax_error = issubclass(self._te.exc_type, SyntaxError)
        if include_title:
            if is_syntax_error:
                title = 'Syntax Error'
            else:
                title = 'Traceback <em>(most recent call last)</em>:'
        else:
            title = ''
        exc_full = escape(''.join(self._te.format_exception_only()))
        if is_syntax_error:
            description = f'<pre class=syntaxerror>{exc_full}</pre>'
        else:
            description = f'<blockquote>{exc_full}</blockquote>'
        return SUMMARY_HTML % {'classes': classes, 'title': f'<h3>{title}</h3>', 'frames': '\n'.join(rows), 'description': description}

    def render_debugger_html(self, evalex: bool, secret: str, evalex_trusted: bool) -> str:
        exc_lines = list(self._te.format_exception_only())
        plaintext = ''.join(self._te.format())
        return PAGE_HTML % {'evalex': 'true' if evalex else 'false', 'evalex_trusted': 'true' if evalex_trusted else 'false', 'console': 'false', 'title': escape(exc_lines[0]), 'exception': escape(''.join(exc_lines)), 'exception_type': escape(self._te.exc_type.__name__), 'summary': self.render_traceback_html(include_title=False), 'plaintext': escape(plaintext), 'plaintext_cs': re.sub('-{2,}', '-', plaintext), 'secret': secret}