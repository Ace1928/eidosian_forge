import re
from functools import partial, reduce
from math import gcd
from operator import itemgetter
from typing import (
from ._loop import loop_last
from ._pick import pick_bool
from ._wrap import divide_line
from .align import AlignMethod
from .cells import cell_len, set_cell_size
from .containers import Lines
from .control import strip_control_codes
from .emoji import EmojiVariant
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import Style, StyleType
def with_indent_guides(self, indent_size: Optional[int]=None, *, character: str='│', style: StyleType='dim green') -> 'Text':
    """Adds indent guide lines to text.

        Args:
            indent_size (Optional[int]): Size of indentation, or None to auto detect. Defaults to None.
            character (str, optional): Character to use for indentation. Defaults to "│".
            style (Union[Style, str], optional): Style of indent guides.

        Returns:
            Text: New text with indentation guides.
        """
    _indent_size = self.detect_indentation() if indent_size is None else indent_size
    text = self.copy()
    text.expand_tabs()
    indent_line = f'{character}{' ' * (_indent_size - 1)}'
    re_indent = re.compile('^( *)(.*)$')
    new_lines: List[Text] = []
    add_line = new_lines.append
    blank_lines = 0
    for line in text.split(allow_blank=True):
        match = re_indent.match(line.plain)
        if not match or not match.group(2):
            blank_lines += 1
            continue
        indent = match.group(1)
        full_indents, remaining_space = divmod(len(indent), _indent_size)
        new_indent = f'{indent_line * full_indents}{' ' * remaining_space}'
        line.plain = new_indent + line.plain[len(new_indent):]
        line.stylize(style, 0, len(new_indent))
        if blank_lines:
            new_lines.extend([Text(new_indent, style=style)] * blank_lines)
            blank_lines = 0
        add_line(line)
    if blank_lines:
        new_lines.extend([Text('', style=style)] * blank_lines)
    new_text = text.blank_copy('\n').join(new_lines)
    return new_text