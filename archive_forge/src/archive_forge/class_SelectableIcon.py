from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from urwid.command_map import Command
from urwid.signals import connect_signal
from urwid.text_layout import calc_coords
from urwid.util import is_mouse_press
from .columns import Columns
from .constants import Align, WrapMode
from .text import Text
from .widget import WidgetError, WidgetWrap
class SelectableIcon(Text):
    ignore_focus = False
    _selectable = True

    def __init__(self, text: str | tuple[Hashable, str] | list[str | tuple[Hashable, str]], cursor_position: int=0, align: Literal['left', 'center', 'right'] | Align=Align.LEFT, wrap: Literal['space', 'any', 'clip', 'ellipsis'] | WrapMode=WrapMode.SPACE, layout: TextLayout | None=None) -> None:
        """
        :param text: markup for this widget; see :class:`Text` for
                     description of text markup
        :param cursor_position: position the cursor will appear in the
                                text when this widget is in focus
        :param align: typically ``'left'``, ``'center'`` or ``'right'``
        :type align: text alignment mode
        :param wrap: typically ``'space'``, ``'any'``, ``'clip'`` or ``'ellipsis'``
        :type wrap: text wrapping mode
        :param layout: defaults to a shared :class:`StandardTextLayout` instance
        :type layout: text layout instance

        This is a text widget that is selectable.  A cursor
        displayed at a fixed location in the text when in focus.
        This widget has no special handling of keyboard or mouse input.
        """
        super().__init__(text, align=align, wrap=wrap, layout=layout)
        self._cursor_position = cursor_position

    def render(self, size: tuple[int] | tuple[()], focus: bool=False) -> TextCanvas | CompositeCanvas:
        """
        Render the text content of this widget with a cursor when
        in focus.

        >>> si = SelectableIcon(u"[!]")
        >>> si
        <SelectableIcon selectable fixed/flow widget '[!]'>
        >>> si.render((4,), focus=True).cursor
        (0, 0)
        >>> si = SelectableIcon("((*))", 2)
        >>> si.render((8,), focus=True).cursor
        (2, 0)
        >>> si.render((2,), focus=True).cursor
        (0, 1)
        >>> si.render(()).cursor
        >>> si.render(()).text
        [b'((*))']
        >>> si.render((), focus=True).cursor
        (2, 0)
        """
        c: TextCanvas | CompositeCanvas = super().render(size, focus)
        if focus:
            c = CompositeCanvas(c)
            c.cursor = self.get_cursor_coords(size)
        return c

    def get_cursor_coords(self, size: tuple[int] | tuple[()]) -> tuple[int, int] | None:
        """
        Return the position of the cursor if visible.  This method
        is required for widgets that display a cursor.
        """
        if self._cursor_position > len(self.text):
            return None
        if size:
            maxcol, = size
        else:
            maxcol, _ = self.pack()
        trans = self.get_line_translation(maxcol)
        x, y = calc_coords(self.text, trans, self._cursor_position)
        if maxcol <= x:
            return None
        return (x, y)

    def keypress(self, size: tuple[int] | tuple[()], key: str) -> str:
        """
        No keys are handled by this widget.  This method is
        required for selectable widgets.
        """
        return key