from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from .controls import UIControl, TokenListControl, UIContent
from .dimension import LayoutDimension, sum_layout_dimensions, max_layout_dimensions
from .margins import Margin
from .screen import Point, WritePosition, _CHAR_CACHE
from .utils import token_list_to_text, explode_tokens
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import to_cli_filter, ViInsertMode, EmacsInsertMode
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from prompt_toolkit.utils import take_using_weights, get_cwidth
class FloatContainer(Container):
    """
    Container which can contain another container for the background, as well
    as a list of floating containers on top of it.

    Example Usage::

        FloatContainer(content=Window(...),
                       floats=[
                           Float(xcursor=True,
                                ycursor=True,
                                layout=CompletionMenu(...))
                       ])
    """

    def __init__(self, content, floats):
        assert isinstance(content, Container)
        assert all((isinstance(f, Float) for f in floats))
        self.content = content
        self.floats = floats

    def reset(self):
        self.content.reset()
        for f in self.floats:
            f.content.reset()

    def preferred_width(self, cli, write_position):
        return self.content.preferred_width(cli, write_position)

    def preferred_height(self, cli, width, max_available_height):
        """
        Return the preferred height of the float container.
        (We don't care about the height of the floats, they should always fit
        into the dimensions provided by the container.)
        """
        return self.content.preferred_height(cli, width, max_available_height)

    def write_to_screen(self, cli, screen, mouse_handlers, write_position):
        self.content.write_to_screen(cli, screen, mouse_handlers, write_position)
        for fl in self.floats:
            cursor_position = screen.menu_position or screen.cursor_position
            cursor_position = Point(x=cursor_position.x - write_position.xpos, y=cursor_position.y - write_position.ypos)
            fl_width = fl.get_width(cli)
            fl_height = fl.get_height(cli)
            if fl.left is not None and fl_width is not None:
                xpos = fl.left
                width = fl_width
            elif fl.left is not None and fl.right is not None:
                xpos = fl.left
                width = write_position.width - fl.left - fl.right
            elif fl_width is not None and fl.right is not None:
                xpos = write_position.width - fl.right - fl_width
                width = fl_width
            elif fl.xcursor:
                width = fl_width
                if width is None:
                    width = fl.content.preferred_width(cli, write_position.width).preferred
                    width = min(write_position.width, width)
                xpos = cursor_position.x
                if xpos + width > write_position.width:
                    xpos = max(0, write_position.width - width)
            elif fl_width:
                xpos = int((write_position.width - fl_width) / 2)
                width = fl_width
            else:
                width = fl.content.preferred_width(cli, write_position.width).preferred
                if fl.left is not None:
                    xpos = fl.left
                elif fl.right is not None:
                    xpos = max(0, write_position.width - width - fl.right)
                else:
                    xpos = max(0, int((write_position.width - width) / 2))
                width = min(width, write_position.width - xpos)
            if fl.top is not None and fl_height is not None:
                ypos = fl.top
                height = fl_height
            elif fl.top is not None and fl.bottom is not None:
                ypos = fl.top
                height = write_position.height - fl.top - fl.bottom
            elif fl_height is not None and fl.bottom is not None:
                ypos = write_position.height - fl_height - fl.bottom
                height = fl_height
            elif fl.ycursor:
                ypos = cursor_position.y + 1
                height = fl_height
                if height is None:
                    height = fl.content.preferred_height(cli, width, write_position.extended_height).preferred
                if height > write_position.extended_height - ypos:
                    if write_position.extended_height - ypos + 1 >= ypos:
                        height = write_position.extended_height - ypos
                    else:
                        height = min(height, cursor_position.y)
                        ypos = cursor_position.y - height
            elif fl_width:
                ypos = int((write_position.height - fl_height) / 2)
                height = fl_height
            else:
                height = fl.content.preferred_height(cli, width, write_position.extended_height).preferred
                if fl.top is not None:
                    ypos = fl.top
                elif fl.bottom is not None:
                    ypos = max(0, write_position.height - height - fl.bottom)
                else:
                    ypos = max(0, int((write_position.height - height) / 2))
                height = min(height, write_position.height - ypos)
            if height > 0 and width > 0:
                wp = WritePosition(xpos=xpos + write_position.xpos, ypos=ypos + write_position.ypos, width=width, height=height)
                if not fl.hide_when_covering_content or self._area_is_empty(screen, wp):
                    fl.content.write_to_screen(cli, screen, mouse_handlers, wp)

    def _area_is_empty(self, screen, write_position):
        """
        Return True when the area below the write position is still empty.
        (For floats that should not hide content underneath.)
        """
        wp = write_position
        Transparent = Token.Transparent
        for y in range(wp.ypos, wp.ypos + wp.height):
            if y in screen.data_buffer:
                row = screen.data_buffer[y]
                for x in range(wp.xpos, wp.xpos + wp.width):
                    c = row[x]
                    if c.char != ' ' or c.token != Transparent:
                        return False
        return True

    def walk(self, cli):
        """ Walk through children. """
        yield self
        for i in self.content.walk(cli):
            yield i
        for f in self.floats:
            for i in f.content.walk(cli):
                yield i