from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .utils import token_list_to_text
class NumberredMargin(Margin):
    """
    Margin that displays the line numbers.

    :param relative: Number relative to the cursor position. Similar to the Vi
                     'relativenumber' option.
    :param display_tildes: Display tildes after the end of the document, just
        like Vi does.
    """

    def __init__(self, relative=False, display_tildes=False):
        self.relative = to_cli_filter(relative)
        self.display_tildes = to_cli_filter(display_tildes)

    def get_width(self, cli, get_ui_content):
        line_count = get_ui_content().line_count
        return max(3, len('%s' % line_count) + 1)

    def create_margin(self, cli, window_render_info, width, height):
        relative = self.relative(cli)
        token = Token.LineNumber
        token_current = Token.LineNumber.Current
        current_lineno = window_render_info.ui_content.cursor_position.y
        result = []
        last_lineno = None
        for y, lineno in enumerate(window_render_info.displayed_lines):
            if lineno != last_lineno:
                if lineno is None:
                    pass
                elif lineno == current_lineno:
                    if relative:
                        result.append((token_current, '%i' % (lineno + 1)))
                    else:
                        result.append((token_current, ('%i ' % (lineno + 1)).rjust(width)))
                else:
                    if relative:
                        lineno = abs(lineno - current_lineno) - 1
                    result.append((token, ('%i ' % (lineno + 1)).rjust(width)))
            last_lineno = lineno
            result.append((Token, '\n'))
        if self.display_tildes(cli):
            while y < window_render_info.window_height:
                result.append((Token.Tilde, '~\n'))
                y += 1
        return result