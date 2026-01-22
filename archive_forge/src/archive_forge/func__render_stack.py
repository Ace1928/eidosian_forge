from __future__ import absolute_import
import linecache
import os
import platform
import sys
from dataclasses import dataclass, field
from traceback import walk_tb
from types import ModuleType, TracebackType
from typing import (
from pip._vendor.pygments.lexers import guess_lexer_for_filename
from pip._vendor.pygments.token import Comment, Keyword, Name, Number, Operator, String
from pip._vendor.pygments.token import Text as TextToken
from pip._vendor.pygments.token import Token
from pip._vendor.pygments.util import ClassNotFound
from . import pretty
from ._loop import loop_last
from .columns import Columns
from .console import Console, ConsoleOptions, ConsoleRenderable, RenderResult, group
from .constrain import Constrain
from .highlighter import RegexHighlighter, ReprHighlighter
from .panel import Panel
from .scope import render_scope
from .style import Style
from .syntax import Syntax
from .text import Text
from .theme import Theme
@group()
def _render_stack(self, stack: Stack) -> RenderResult:
    path_highlighter = PathHighlighter()
    theme = self.theme

    def read_code(filename: str) -> str:
        """Read files, and cache results on filename.

            Args:
                filename (str): Filename to read

            Returns:
                str: Contents of file
            """
        return ''.join(linecache.getlines(filename))

    def render_locals(frame: Frame) -> Iterable[ConsoleRenderable]:
        if frame.locals:
            yield render_scope(frame.locals, title='locals', indent_guides=self.indent_guides, max_length=self.locals_max_length, max_string=self.locals_max_string)
    exclude_frames: Optional[range] = None
    if self.max_frames != 0:
        exclude_frames = range(self.max_frames // 2, len(stack.frames) - self.max_frames // 2)
    excluded = False
    for frame_index, frame in enumerate(stack.frames):
        if exclude_frames and frame_index in exclude_frames:
            excluded = True
            continue
        if excluded:
            assert exclude_frames is not None
            yield Text(f'\n... {len(exclude_frames)} frames hidden ...', justify='center', style='traceback.error')
            excluded = False
        first = frame_index == 0
        frame_filename = frame.filename
        suppressed = any((frame_filename.startswith(path) for path in self.suppress))
        if os.path.exists(frame.filename):
            text = Text.assemble(path_highlighter(Text(frame.filename, style='pygments.string')), (':', 'pygments.text'), (str(frame.lineno), 'pygments.number'), ' in ', (frame.name, 'pygments.function'), style='pygments.text')
        else:
            text = Text.assemble('in ', (frame.name, 'pygments.function'), (':', 'pygments.text'), (str(frame.lineno), 'pygments.number'), style='pygments.text')
        if not frame.filename.startswith('<') and (not first):
            yield ''
        yield text
        if frame.filename.startswith('<'):
            yield from render_locals(frame)
            continue
        if not suppressed:
            try:
                code = read_code(frame.filename)
                if not code:
                    continue
                lexer_name = self._guess_lexer(frame.filename, code)
                syntax = Syntax(code, lexer_name, theme=theme, line_numbers=True, line_range=(frame.lineno - self.extra_lines, frame.lineno + self.extra_lines), highlight_lines={frame.lineno}, word_wrap=self.word_wrap, code_width=88, indent_guides=self.indent_guides, dedent=False)
                yield ''
            except Exception as error:
                yield Text.assemble((f'\n{error}', 'traceback.error'))
            else:
                yield (Columns([syntax, *render_locals(frame)], padding=1) if frame.locals else syntax)