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
@dataclass
class _SyntaxError:
    offset: int
    filename: str
    line: str
    lineno: int
    msg: str