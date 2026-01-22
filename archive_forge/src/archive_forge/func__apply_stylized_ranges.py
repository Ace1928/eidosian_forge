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
def _apply_stylized_ranges(self, text: Text) -> None:
    """
        Apply stylized ranges to a text instance,
        using the given code to determine the right portion to apply the style to.

        Args:
            text (Text): Text instance to apply the style to.
        """
    code = text.plain
    newlines_offsets = [0, *[match.start() + 1 for match in re.finditer('\n', code, flags=re.MULTILINE)], len(code) + 1]
    for stylized_range in self._stylized_ranges:
        start = _get_code_index_for_syntax_position(newlines_offsets, stylized_range.start)
        end = _get_code_index_for_syntax_position(newlines_offsets, stylized_range.end)
        if start is not None and end is not None:
            text.stylize(stylized_range.style, start, end)