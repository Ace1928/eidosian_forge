import os.path
import platform
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from pip._vendor.pygments.lexer import Lexer
from pip._vendor.pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pip._vendor.pygments.style import Style as PygmentsStyle
from pip._vendor.pygments.styles import get_style_by_name
from pip._vendor.pygments.token import (
from pip._vendor.pygments.util import ClassNotFound
from pip._vendor.rich.containers import Lines
from pip._vendor.rich.padding import Padding, PaddingDimensions
from ._loop import loop_first
from .cells import cell_len
from .color import Color, blend_rgb
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment, Segments
from .style import Style, StyleType
from .text import Text
def tokens_to_spans() -> Iterable[Tuple[str, Optional[Style]]]:
    """Convert tokens to spans."""
    tokens = iter(line_tokenize())
    line_no = 0
    _line_start = line_start - 1 if line_start else 0
    while line_no < _line_start:
        try:
            _token_type, token = next(tokens)
        except StopIteration:
            break
        yield (token, None)
        if token.endswith('\n'):
            line_no += 1
    for token_type, token in tokens:
        yield (token, _get_theme_style(token_type))
        if token.endswith('\n'):
            line_no += 1
            if line_end and line_no >= line_end:
                break