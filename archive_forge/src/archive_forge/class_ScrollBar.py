from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
class ScrollBar(WidgetDecoration[WrappedWidget]):
    Symbols = ScrollbarSymbols

    def sizing(self) -> frozenset[Sizing]:
        return frozenset((Sizing.BOX,))

    def selectable(self) -> bool:
        return True

    def __init__(self, widget: WrappedWidget, thumb_char: str=ScrollbarSymbols.FULL_BLOCK, trough_char: str=' ', side: Literal['left', 'right']=SCROLLBAR_RIGHT, width: int=1) -> None:
        """Box widget that adds a scrollbar to `widget`

        `widget` must be a box widget with the following methods:
          - `get_scrollpos` takes the arguments `size` and `focus` and returns the index of the first visible row.
          - `set_scrollpos` (optional; needed for mouse click support) takes the index of the first visible row.
          - `rows_max` takes `size` and `focus` and returns the total number of rows `widget` can render.

        `thumb_char` is the character used for the scrollbar handle.
        `trough_char` is used for the space above and below the handle.
        `side` must be 'left' or 'right'.
        `width` specifies the number of columns the scrollbar uses.
        """
        if Sizing.BOX not in widget.sizing():
            raise ValueError(f'Not a box widget: {widget!r}')
        if not isinstance(widget, SupportsScroll):
            raise TypeError(f'Not a scrollable widget: {widget!r}')
        super().__init__(widget)
        self._thumb_char = thumb_char
        self._trough_char = trough_char
        self.scrollbar_side = side
        self.scrollbar_width = max(1, width)
        self._original_widget_size = (0, 0)

    def render(self, size: tuple[int, int], focus: bool=False) -> Canvas:
        from urwid import canvas

        def render_no_scrollbar() -> Canvas:
            self._original_widget_size = size
            return ow.render(size, focus)

        def render_for_scrollbar() -> Canvas:
            self._original_widget_size = ow_size
            return ow.render(ow_size, focus)
        maxcol, maxrow = size
        ow_size = (max(0, maxcol - self._scrollbar_width), maxrow)
        sb_width = maxcol - ow_size[0]
        ow = self._original_widget
        ow_base = self.scrolling_base_widget
        use_relative = isinstance(ow_base, SupportsRelativeScroll) and any((hasattr(ow_base, attrib) for attrib in ('__length_hint__', '__len__'))) and ow_base.require_relative_scroll(size, focus)
        if use_relative:
            ow_len = getattr(ow_base, '__len__', getattr(ow_base, '__length_hint__', lambda: 0))()
            ow_canv = render_for_scrollbar()
            visible_amount = ow_base.get_visible_amount(ow_size, focus)
            pos = ow_base.get_first_visible_pos(ow_size, focus)
            ow_len = max(ow_len, visible_amount, pos)
            posmax = ow_len - visible_amount
            thumb_weight = min(1.0, visible_amount / max(1, ow_len))
            if ow_len == visible_amount:
                use_relative = False
        if not use_relative:
            ow_rows_max = ow_base.rows_max(size, focus)
            if ow_rows_max <= maxrow:
                return render_no_scrollbar()
            ow_canv = render_for_scrollbar()
            ow_rows_max = ow_base.rows_max(ow_size, focus)
            pos = ow_base.get_scrollpos(ow_size, focus)
            posmax = ow_rows_max - maxrow
            thumb_weight = min(1.0, maxrow / max(1, ow_rows_max))
        thumb_height = max(1, round(thumb_weight * maxrow))
        top_weight = float(pos) / max(1, posmax)
        top_height = int((maxrow - thumb_height) * top_weight)
        if top_height == 0 and top_weight > 0:
            top_height = 1
        bottom_height = maxrow - thumb_height - top_height
        top = canvas.SolidCanvas(self._trough_char, sb_width, 1)
        thumb = canvas.SolidCanvas(self._thumb_char, sb_width, 1)
        bottom = canvas.SolidCanvas(self._trough_char, sb_width, 1)
        sb_canv = canvas.CanvasCombine((*((top, None, False) for _ in range(top_height)), *((thumb, None, False) for _ in range(thumb_height)), *((bottom, None, False) for _ in range(bottom_height))))
        combinelist = [(ow_canv, None, True, ow_size[0]), (sb_canv, None, False, sb_width)]
        if self._scrollbar_side != SCROLLBAR_LEFT:
            return canvas.CanvasJoin(combinelist)
        return canvas.CanvasJoin(reversed(combinelist))

    @property
    def scrollbar_width(self) -> int:
        """Columns the scrollbar uses"""
        return max(1, self._scrollbar_width)

    @scrollbar_width.setter
    def scrollbar_width(self, width: typing.SupportsInt) -> None:
        self._scrollbar_width = max(1, int(width))
        self._invalidate()

    @property
    def scrollbar_side(self) -> Literal['left', 'right']:
        """Where to display the scrollbar; must be 'left' or 'right'"""
        return self._scrollbar_side

    @scrollbar_side.setter
    def scrollbar_side(self, side: Literal['left', 'right']) -> None:
        if side not in {SCROLLBAR_LEFT, SCROLLBAR_RIGHT}:
            raise ValueError(f'scrollbar_side must be "left" or "right", not {side!r}')
        self._scrollbar_side = side
        self._invalidate()

    @property
    def scrolling_base_widget(self) -> SupportsScroll | SupportsRelativeScroll:
        """Nearest `original_widget` that is compatible with the scrolling API"""

        def orig_iter(w: Widget) -> Iterator[Widget]:
            while hasattr(w, 'original_widget'):
                w = w.original_widget
                yield w
            yield w
        w = self
        for w in orig_iter(self):
            if isinstance(w, SupportsScroll):
                return w
        raise ScrollableError(f'Not compatible to be wrapped by ScrollBar: {w!r}')

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        return self._original_widget.keypress(self._original_widget_size, key)

    def mouse_event(self, size: tuple[int, int], event: str, button: int, col: int, row: int, focus: bool) -> bool | None:
        ow = self._original_widget
        ow_size = self._original_widget_size
        handled: bool | None = False
        if hasattr(ow, 'mouse_event'):
            handled = ow.mouse_event(ow_size, event, button, col, row, focus)
        if not handled and hasattr(ow, 'set_scrollpos'):
            if button == 4:
                pos = ow.get_scrollpos(ow_size)
                newpos = max(pos - 1, 0)
                ow.set_scrollpos(newpos)
                return True
            if button == 5:
                pos = ow.get_scrollpos(ow_size)
                ow.set_scrollpos(pos + 1)
                return True
        return False