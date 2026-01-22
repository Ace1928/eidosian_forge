from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import (
from prompt_toolkit.utils import get_cwidth
from .controls import UIContent
class NumberedMargin(Margin):
    """
    Margin that displays the line numbers.

    :param relative: Number relative to the cursor position. Similar to the Vi
                     'relativenumber' option.
    :param display_tildes: Display tildes after the end of the document, just
        like Vi does.
    """

    def __init__(self, relative: FilterOrBool=False, display_tildes: FilterOrBool=False) -> None:
        self.relative = to_filter(relative)
        self.display_tildes = to_filter(display_tildes)

    def get_width(self, get_ui_content: Callable[[], UIContent]) -> int:
        line_count = get_ui_content().line_count
        return max(3, len('%s' % line_count) + 1)

    def create_margin(self, window_render_info: WindowRenderInfo, width: int, height: int) -> StyleAndTextTuples:
        relative = self.relative()
        style = 'class:line-number'
        style_current = 'class:line-number.current'
        current_lineno = window_render_info.ui_content.cursor_position.y
        result: StyleAndTextTuples = []
        last_lineno = None
        for y, lineno in enumerate(window_render_info.displayed_lines):
            if lineno != last_lineno:
                if lineno is None:
                    pass
                elif lineno == current_lineno:
                    if relative:
                        result.append((style_current, '%i' % (lineno + 1)))
                    else:
                        result.append((style_current, ('%i ' % (lineno + 1)).rjust(width)))
                else:
                    if relative:
                        lineno = abs(lineno - current_lineno) - 1
                    result.append((style, ('%i ' % (lineno + 1)).rjust(width)))
            last_lineno = lineno
            result.append(('', '\n'))
        if self.display_tildes():
            while y < window_render_info.window_height:
                result.append(('class:tilde', '~\n'))
                y += 1
        return result