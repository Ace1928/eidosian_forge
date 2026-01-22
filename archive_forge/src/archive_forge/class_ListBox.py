from __future__ import annotations
import operator
import typing
import warnings
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing_extensions import Protocol, runtime_checkable
from urwid import signals
from urwid.canvas import CanvasCombine, SolidCanvas
from .constants import Sizing, VAlign, WHSettings, normalize_valign
from .container import WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, nocache_widget_render_instance
class ListBox(Widget, WidgetContainerMixin):
    """
    Vertically stacked list of widgets
    """
    _selectable = True
    _sizing = frozenset([Sizing.BOX])

    def __init__(self, body: ListWalker | Iterable[Widget]) -> None:
        """
        :param body: a ListWalker subclass such as :class:`SimpleFocusListWalker`
            that contains widgets to be displayed inside the list box
        :type body: ListWalker
        """
        super().__init__()
        if getattr(body, 'get_focus', None):
            self._body: ListWalker = body
        else:
            self._body = SimpleListWalker(body)
        self.body = self._body
        self.offset_rows = 0
        self.inset_fraction = (0, 1)
        self.pref_col = 'left'
        self.set_focus_pending = 'first selectable'
        self.set_focus_valign_pending = None
        self._rows_max_cached = 0
        self._rendered_size = (0, 0)

    @property
    def body(self) -> ListWalker:
        """
        a ListWalker subclass such as :class:`SimpleFocusListWalker` that contains
        widgets to be displayed inside the list box
        """
        return self._body

    @body.setter
    def body(self, body: Iterable[Widget] | ListWalker) -> None:
        with suppress(AttributeError):
            signals.disconnect_signal(self._body, 'modified', self._invalidate)
        if getattr(body, 'get_focus', None):
            self._body = body
        else:
            self._body = SimpleListWalker(body)
        try:
            signals.connect_signal(self._body, 'modified', self._invalidate)
        except NameError:
            self.render = nocache_widget_render_instance(self)
        self._invalidate()

    def _get_body(self):
        warnings.warn(f'Method `{self.__class__.__name__}._get_body` is deprecated, please use property `{self.__class__.__name__}.body`', DeprecationWarning, stacklevel=3)
        return self.body

    def _set_body(self, body):
        warnings.warn(f'Method `{self.__class__.__name__}._set_body` is deprecated, please use property `{self.__class__.__name__}.body`', DeprecationWarning, stacklevel=3)
        self.body = body

    @property
    def __len__(self) -> Callable[[], int]:
        if isinstance(self._body, Sized):
            return self._body.__len__
        raise AttributeError(f'{self._body.__class__.__name__} is not Sized')

    @property
    def __length_hint__(self) -> Callable[[], int]:
        if isinstance(self._body, (Sized, EstimatedSized)):
            return lambda: operator.length_hint(self._body)
        raise AttributeError(f'{self._body.__class__.__name__} is not Sized and do not implement "__length_hint__"')

    def calculate_visible(self, size: tuple[int, int], focus: bool=False) -> VisibleInfo | tuple[None, None, None]:
        """
        Returns the widgets that would be displayed in
        the ListBox given the current *size* and *focus*.

        see :meth:`Widget.render` for parameter details

        :returns: (*middle*, *top*, *bottom*) or (``None``, ``None``, ``None``)

        *middle*
            (*row offset*(when +ve) or *inset*(when -ve),
            *focus widget*, *focus position*, *focus rows*,
            *cursor coords* or ``None``)
        *top*
            (*# lines to trim off top*,
            list of (*widget*, *position*, *rows*) tuples above focus in order from bottom to top)
        *bottom*
            (*# lines to trim off bottom*,
            list of (*widget*, *position*, *rows*) tuples below focus in order from top to bottom)
        """
        maxcol, maxrow = size
        if self.set_focus_pending or self.set_focus_valign_pending:
            self._set_focus_complete((maxcol, maxrow), focus)
        focus_widget, focus_pos = self._body.get_focus()
        if focus_widget is None:
            return (None, None, None)
        top_pos = focus_pos
        offset_rows, inset_rows = self.get_focus_offset_inset((maxcol, maxrow))
        if maxrow and offset_rows >= maxrow:
            offset_rows = maxrow - 1
        cursor = None
        if maxrow and focus_widget.selectable() and focus and hasattr(focus_widget, 'get_cursor_coords'):
            cursor = focus_widget.get_cursor_coords((maxcol,))
        if cursor is not None:
            _cx, cy = cursor
            effective_cy = cy + offset_rows - inset_rows
            if effective_cy < 0:
                inset_rows = cy
            elif effective_cy >= maxrow:
                offset_rows = maxrow - cy - 1
                if offset_rows < 0:
                    inset_rows, offset_rows = (-offset_rows, 0)
        trim_top = inset_rows
        focus_rows = focus_widget.rows((maxcol,), True)
        pos = focus_pos
        fill_lines = offset_rows
        fill_above = []
        top_pos = pos
        while fill_lines > 0:
            prev, pos = self._body.get_prev(pos)
            if prev is None:
                offset_rows -= fill_lines
                break
            top_pos = pos
            p_rows = prev.rows((maxcol,))
            if p_rows:
                fill_above.append(VisibleInfoFillItem(prev, pos, p_rows))
            if p_rows > fill_lines:
                trim_top = p_rows - fill_lines
                break
            fill_lines -= p_rows
        trim_bottom = max(focus_rows + offset_rows - inset_rows - maxrow, 0)
        pos = focus_pos
        fill_lines = maxrow - focus_rows - offset_rows + inset_rows
        fill_below = []
        while fill_lines > 0:
            next_pos, pos = self._body.get_next(pos)
            if next_pos is None:
                break
            n_rows = next_pos.rows((maxcol,))
            if n_rows:
                fill_below.append(VisibleInfoFillItem(next_pos, pos, n_rows))
            if n_rows > fill_lines:
                trim_bottom = n_rows - fill_lines
                fill_lines -= n_rows
                break
            fill_lines -= n_rows
        fill_lines = max(0, fill_lines)
        if fill_lines > 0 and trim_top > 0:
            if fill_lines <= trim_top:
                trim_top -= fill_lines
                offset_rows += fill_lines
                fill_lines = 0
            else:
                fill_lines -= trim_top
                offset_rows += trim_top
                trim_top = 0
        pos = top_pos
        while fill_lines > 0:
            prev, pos = self._body.get_prev(pos)
            if prev is None:
                break
            p_rows = prev.rows((maxcol,))
            fill_above.append(VisibleInfoFillItem(prev, pos, p_rows))
            if p_rows > fill_lines:
                trim_top = p_rows - fill_lines
                offset_rows += fill_lines
                break
            fill_lines -= p_rows
            offset_rows += p_rows
        return VisibleInfo(VisibleInfoMiddle(offset_rows - inset_rows, focus_widget, focus_pos, focus_rows, cursor), VisibleInfoTopBottom(trim_top, fill_above), VisibleInfoTopBottom(trim_bottom, fill_below))

    def _check_support_scrolling(self) -> None:
        from .treetools import TreeWalker
        if not isinstance(self._body, ScrollSupportingBody):
            raise ListBoxError(f'{self} body do not implement methods required for scrolling protocol')
        if not isinstance(self._body, (Sized, EstimatedSized, TreeWalker)):
            raise ListBoxError(f"{self} body is not a Sized, can not estimate it's size and not a TreeWalker.Scroll is not allowed due to risk of infinite cycle of widgets load.")
        if getattr(self._body, 'wrap_around', False):
            raise ListBoxError('Body is wrapped around. Scroll position calculation is undefined.')

    def get_scrollpos(self, size: tuple[int, int] | None=None, focus: bool=False) -> int:
        """Current scrolling position."""
        self._check_support_scrolling()
        if not self._body:
            return 0
        if size is not None:
            self._rendered_size = size
        mid, top, _bottom = self.calculate_visible(self._rendered_size, focus)
        start_row = top.trim
        maxcol = self._rendered_size[0]
        if top.fill:
            pos = top.fill[-1].position
        else:
            pos = mid.focus_pos
        prev, pos = self._body.get_prev(pos)
        while prev is not None:
            start_row += prev.rows((maxcol,))
            prev, pos = self._body.get_prev(pos)
        return start_row

    def rows_max(self, size: tuple[int, int] | None=None, focus: bool=False) -> int:
        """Scrollable protocol for sized iterable and not wrapped around contents."""
        self._check_support_scrolling()
        if size is not None:
            self._rendered_size = size
        if size or not self._rows_max_cached:
            cols = self._rendered_size[0]
            rows = 0
            focused_w, idx = self.body.get_focus()
            rows += focused_w.rows((cols,), focus)
            prev, pos = self._body.get_prev(idx)
            while prev is not None:
                rows += prev.rows((cols,), False)
                prev, pos = self._body.get_prev(pos)
            next_, pos = self.body.get_next(idx)
            while next_ is not None:
                rows += next_.rows((cols,), True)
                next_, pos = self._body.get_next(pos)
            self._rows_max_cached = rows
        return self._rows_max_cached

    def require_relative_scroll(self, size: tuple[int, int], focus: bool=False) -> bool:
        """Widget require relative scroll due to performance limitations of real lines count calculation."""
        return isinstance(self._body, (Sized, EstimatedSized)) and size[1] * 3 < operator.length_hint(self.body)

    def get_first_visible_pos(self, size: tuple[int, int], focus: bool=False) -> int:
        self._check_support_scrolling()
        if not self._body:
            return 0
        _mid, top, _bottom = self.calculate_visible(size, focus)
        if top.fill:
            first_pos = top.fill[-1].position
        else:
            first_pos = self.focus_position
        over = 0
        _widget, first_pos = self.body.get_prev(first_pos)
        while first_pos is not None:
            over += 1
            _widget, first_pos = self.body.get_prev(first_pos)
        return over

    def get_visible_amount(self, size: tuple[int, int], focus: bool=False) -> int:
        self._check_support_scrolling()
        if not self._body:
            return 1
        _mid, top, bottom = self.calculate_visible(size, focus)
        return 1 + len(top.fill) + len(bottom.fill)

    def render(self, size: tuple[int, int], focus: bool=False) -> CompositeCanvas | SolidCanvas:
        """
        Render ListBox and return canvas.

        see :meth:`Widget.render` for details
        """
        maxcol, maxrow = size
        self._rendered_size = size
        middle, top, bottom = self.calculate_visible((maxcol, maxrow), focus=focus)
        if middle is None:
            return SolidCanvas(' ', maxcol, maxrow)
        _ignore, focus_widget, focus_pos, focus_rows, cursor = middle
        trim_top, fill_above = top
        trim_bottom, fill_below = bottom
        combinelist: list[tuple[Canvas, int, bool]] = []
        rows = 0
        fill_above.reverse()
        for widget, w_pos, w_rows in fill_above:
            canvas = widget.render((maxcol,))
            if w_rows != canvas.rows():
                raise ListBoxError(f'Widget {widget!r} at position {w_pos!r} within listbox calculated {w_rows:d} rows but rendered {canvas.rows():d}!')
            rows += w_rows
            combinelist.append((canvas, w_pos, False))
        focus_canvas = focus_widget.render((maxcol,), focus=focus)
        if focus_canvas.rows() != focus_rows:
            raise ListBoxError(f'Focus Widget {focus_widget!r} at position {focus_pos!r} within listbox calculated {focus_rows:d} rows but rendered {focus_canvas.rows():d}!')
        c_cursor = focus_canvas.cursor
        if cursor is not None and cursor != c_cursor:
            raise ListBoxError(f'Focus Widget {focus_widget!r} at position {focus_pos!r} within listbox calculated cursor coords {cursor!r} but rendered cursor coords {c_cursor!r}!')
        rows += focus_rows
        combinelist.append((focus_canvas, focus_pos, True))
        for widget, w_pos, w_rows in fill_below:
            canvas = widget.render((maxcol,))
            if w_rows != canvas.rows():
                raise ListBoxError(f'Widget {widget!r} at position {w_pos!r} within listbox calculated {w_rows:d} rows but rendered {canvas.rows():d}!')
            rows += w_rows
            combinelist.append((canvas, w_pos, False))
        final_canvas = CanvasCombine(combinelist)
        if trim_top:
            final_canvas.trim(trim_top)
            rows -= trim_top
        if trim_bottom:
            final_canvas.trim_end(trim_bottom)
            rows -= trim_bottom
        if rows > maxrow:
            raise ListBoxError(f'Listbox contents too long!\nRender top={top!r}, middle={middle!r}, bottom={bottom!r}\n')
        if rows < maxrow:
            if trim_bottom != 0:
                raise ListBoxError(f'Listbox contents too short!\nRender top={top!r}, middle={middle!r}, bottom={bottom!r}\nTrim bottom={trim_bottom!r}')
            bottom_pos = focus_pos
            if fill_below:
                bottom_pos = fill_below[-1][1]
            rendered_positions = frozenset((idx for _, idx, _ in combinelist))
            widget, next_pos = self._body.get_next(bottom_pos)
            while all((widget is not None, next_pos is not None, next_pos not in rendered_positions)):
                if widget.rows((maxcol,), False):
                    raise ListBoxError(f'Listbox contents too short!\nRender top={top!r}, middle={middle!r}, bottom={bottom!r}\nNot rendered not empty widgets available (first is {widget!r} with position {next_pos!r})')
                widget, next_next_pos = self._body.get_next(next_pos)
                if next_pos == next_next_pos:
                    raise ListBoxError(f'Next position after {next_pos!r} is invalid (points to itself)\nLooks like bug with {self._body!r}')
                next_pos = next_next_pos
            final_canvas.pad_trim_top_bottom(0, maxrow - rows)
        return final_canvas

    def get_cursor_coords(self, size: tuple[int, int]) -> tuple[int, int] | None:
        """
        See :meth:`Widget.get_cursor_coords` for details
        """
        maxcol, maxrow = size
        middle, _top, _bottom = self.calculate_visible((maxcol, maxrow), True)
        if middle is None:
            return None
        offset_inset, _ignore1, _ignore2, _ignore3, cursor = middle
        if not cursor:
            return None
        x, y = cursor
        y += offset_inset
        if y < 0 or y >= maxrow:
            return None
        return (x, y)

    def set_focus_valign(self, valign: Literal['top', 'middle', 'bottom'] | VAlign | tuple[Literal['relative', WHSettings.RELATIVE], int]):
        """Set the focus widget's display offset and inset.

        :param valign: one of: 'top', 'middle', 'bottom' ('relative', percentage 0=top 100=bottom)
        """
        vt, va = normalize_valign(valign, ListBoxError)
        self.set_focus_valign_pending = (vt, va)

    def set_focus(self, position, coming_from: Literal['above', 'below'] | None=None) -> None:
        """
        Set the focus position and try to keep the old focus in view.

        :param position: a position compatible with :meth:`self._body.set_focus`
        :param coming_from: set to 'above' or 'below' if you know that
                            old position is above or below the new position.
        :type coming_from: str
        """
        if coming_from not in {'above', 'below', None}:
            raise ListBoxError(f'coming_from value invalid: {coming_from!r}')
        focus_widget, focus_pos = self._body.get_focus()
        if focus_widget is None:
            raise IndexError("Can't set focus, ListBox is empty")
        self.set_focus_pending = (coming_from, focus_widget, focus_pos)
        self._body.set_focus(position)

    def get_focus(self):
        """
        Return a `(focus widget, focus position)` tuple, for backwards
        compatibility. You may also use the new standard container
        properties :attr:`focus` and :attr:`focus_position` to read these values.
        """
        warnings.warn('only for backwards compatibility.You may also use the new standard container property `focus` to get the focus and property `focus_position` to read these values.', PendingDeprecationWarning, stacklevel=2)
        return self._body.get_focus()

    @property
    def focus(self) -> Widget | None:
        """
        the child widget in focus or None when ListBox is empty.

        Return the widget in focus according to our :obj:`list walker <ListWalker>`.
        """
        return self._body.get_focus()[0]

    def _get_focus(self) -> Widget:
        warnings.warn(f'method `{self.__class__.__name__}._get_focus` is deprecated, please use `{self.__class__.__name__}.focus` property', DeprecationWarning, stacklevel=3)
        return self.focus

    def _get_focus_position(self):
        """
        Return the list walker position of the widget in focus. The type
        of value returned depends on the :obj:`list walker <ListWalker>`.

        """
        w, pos = self._body.get_focus()
        if w is None:
            raise IndexError('No focus_position, ListBox is empty')
        return pos
    focus_position = property(_get_focus_position, set_focus, doc='\n        the position of child widget in focus. The valid values for this\n        position depend on the list walker in use.\n        :exc:`IndexError` will be raised by reading this property when the\n        ListBox is empty or setting this property to an invalid position.\n        ')

    def _contents(self):

        class ListBoxContents:
            __getitem__ = self._contents__getitem__
            __len__ = self.__len__

            def __repr__(inner_self) -> str:
                return f'<{inner_self.__class__.__name__} for {self!r} at 0x{id(inner_self):X}>'

            def __call__(inner_self) -> Self:
                warnings.warn('ListBox.contents is a property, not a method', DeprecationWarning, stacklevel=3)
                return inner_self
        return ListBoxContents()

    def _contents__getitem__(self, key):
        if hasattr(self._body, '__getitem__'):
            try:
                return (self._body[key], None)
            except (IndexError, KeyError) as exc:
                raise KeyError(f'ListBox.contents key not found: {key!r}').with_traceback(exc.__traceback__) from exc
        _w, old_focus = self._body.get_focus()
        try:
            self._body.set_focus(key)
            return self._body.get_focus()[0]
        except (IndexError, KeyError) as exc:
            raise KeyError(f'ListBox.contents key not found: {key!r}').with_traceback(exc.__traceback__) from exc
        finally:
            self._body.set_focus(old_focus)

    @property
    def contents(self):
        """
        An object that allows reading widgets from the ListBox's list
        walker as a `(widget, options)` tuple. `None` is currently the only
        value for options.

        .. warning::

            This object may not be used to set or iterate over contents.

            You must use the list walker stored as
            :attr:`.body` to perform manipulation and iteration, if supported.
        """
        return self._contents()

    def options(self):
        """
        There are currently no options for ListBox contents.

        Return None as a placeholder for future options.
        """

    def _set_focus_valign_complete(self, size: tuple[int, int], focus: bool) -> None:
        """Finish setting the offset and inset now that we have have a maxcol & maxrow."""
        maxcol, maxrow = size
        vt, va = self.set_focus_valign_pending
        self.set_focus_valign_pending = None
        self.set_focus_pending = None
        focus_widget, _focus_pos = self._body.get_focus()
        if focus_widget is None:
            return
        rows = focus_widget.rows((maxcol,), focus)
        rtop, _rbot = calculate_top_bottom_filler(maxrow, vt, va, WHSettings.GIVEN, rows, None, 0, 0)
        self.shift_focus((maxcol, maxrow), rtop)

    def _set_focus_first_selectable(self, size: tuple[int, int], focus: bool) -> None:
        """Choose the first visible, selectable widget below the current focus as the focus widget."""
        maxcol, maxrow = size
        self.set_focus_valign_pending = None
        self.set_focus_pending = None
        middle, top, bottom = self.calculate_visible((maxcol, maxrow), focus=focus)
        if middle is None:
            return
        row_offset, focus_widget, _focus_pos, focus_rows, _cursor = middle
        _trim_top, _fill_above = top
        trim_bottom, fill_below = bottom
        if focus_widget.selectable():
            return
        if trim_bottom:
            fill_below = fill_below[:-1]
        new_row_offset = row_offset + focus_rows
        for widget, pos, rows in fill_below:
            if widget.selectable():
                self._body.set_focus(pos)
                self.shift_focus((maxcol, maxrow), new_row_offset)
                return
            new_row_offset += rows

    def _set_focus_complete(self, size: tuple[int, int], focus: bool) -> None:
        """Finish setting the position now that we have maxcol & maxrow."""
        maxcol, maxrow = size
        self._invalidate()
        if self.set_focus_pending == 'first selectable':
            return self._set_focus_first_selectable((maxcol, maxrow), focus)
        if self.set_focus_valign_pending is not None:
            return self._set_focus_valign_complete((maxcol, maxrow), focus)
        coming_from, _focus_widget, focus_pos = self.set_focus_pending
        self.set_focus_pending = None
        _new_focus_widget, position = self._body.get_focus()
        if focus_pos == position:
            return None
        self._body.set_focus(focus_pos)
        middle, top, bottom = self.calculate_visible((maxcol, maxrow), focus)
        focus_offset, _focus_widget, focus_pos, focus_rows, _cursor = middle
        _trim_top, fill_above = top
        _trim_bottom, fill_below = bottom
        offset = focus_offset
        for _widget, pos, rows in fill_above:
            offset -= rows
            if pos == position:
                self.change_focus((maxcol, maxrow), pos, offset, 'below')
                return None
        offset = focus_offset + focus_rows
        for _widget, pos, rows in fill_below:
            if pos == position:
                self.change_focus((maxcol, maxrow), pos, offset, 'above')
                return None
            offset += rows
        self._body.set_focus(position)
        widget, position = self._body.get_focus()
        rows = widget.rows((maxcol,), focus)
        if coming_from == 'below':
            offset = 0
        elif coming_from == 'above':
            offset = maxrow - rows
        else:
            offset = (maxrow - rows) // 2
        self.shift_focus((maxcol, maxrow), offset)
        return None

    def shift_focus(self, size: tuple[int, int], offset_inset: int) -> None:
        """
        Move the location of the current focus relative to the top.
        This is used internally by methods that know the widget's *size*.

        See also :meth:`.set_focus_valign`.

        :param size: see :meth:`Widget.render` for details
        :param offset_inset: either the number of rows between the
            top of the listbox and the start of the focus widget (+ve
            value) or the number of lines of the focus widget hidden off
            the top edge of the listbox (-ve value) or ``0`` if the top edge
            of the focus widget is aligned with the top edge of the
            listbox.
        :type offset_inset: int
        """
        maxcol, maxrow = size
        if offset_inset >= 0:
            if offset_inset >= maxrow:
                raise ListBoxError(f'Invalid offset_inset: {offset_inset!r}, only {maxrow!r} rows in list box')
            self.offset_rows = offset_inset
            self.inset_fraction = (0, 1)
        else:
            target, _ignore = self._body.get_focus()
            tgt_rows = target.rows((maxcol,), True)
            if offset_inset + tgt_rows <= 0:
                raise ListBoxError(f'Invalid offset_inset: {offset_inset!r}, only {tgt_rows!r} rows in target!')
            self.offset_rows = 0
            self.inset_fraction = (-offset_inset, tgt_rows)
        self._invalidate()

    def update_pref_col_from_focus(self, size: tuple[int, int]) -> None:
        """Update self.pref_col from the focus widget."""
        maxcol, _maxrow = size
        widget, _old_pos = self._body.get_focus()
        if widget is None:
            return
        pref_col = None
        if hasattr(widget, 'get_pref_col'):
            pref_col = widget.get_pref_col((maxcol,))
        if pref_col is None and hasattr(widget, 'get_cursor_coords'):
            coords = widget.get_cursor_coords((maxcol,))
            if isinstance(coords, tuple):
                pref_col, _y = coords
        if pref_col is not None:
            self.pref_col = pref_col

    def change_focus(self, size: tuple[int, int], position, offset_inset: int=0, coming_from: Literal['above', 'below'] | None=None, cursor_coords: tuple[int, int] | None=None, snap_rows: int | None=None) -> None:
        """
        Change the current focus widget.
        This is used internally by methods that know the widget's *size*.

        See also :meth:`.set_focus`.

        :param size: see :meth:`Widget.render` for details
        :param position: a position compatible with :meth:`self._body.set_focus`
        :param offset_inset: either the number of rows between the
            top of the listbox and the start of the focus widget (+ve
            value) or the number of lines of the focus widget hidden off
            the top edge of the listbox (-ve value) or 0 if the top edge
            of the focus widget is aligned with the top edge of the
            listbox (default if unspecified)
        :type offset_inset: int
        :param coming_from: either 'above', 'below' or unspecified `None`
        :type coming_from: str
        :param cursor_coords: (x, y) tuple indicating the desired
            column and row for the cursor, a (x,) tuple indicating only
            the column for the cursor, or unspecified
        :type cursor_coords: (int, int)
        :param snap_rows: the maximum number of extra rows to scroll
            when trying to "snap" a selectable focus into the view
        :type snap_rows: int
        """
        maxcol, maxrow = size
        if cursor_coords:
            self.pref_col = cursor_coords[0]
        else:
            self.update_pref_col_from_focus((maxcol, maxrow))
        self._invalidate()
        self._body.set_focus(position)
        target, _ignore = self._body.get_focus()
        tgt_rows = target.rows((maxcol,), True)
        if snap_rows is None:
            snap_rows = maxrow - 1
        align_top = 0
        align_bottom = maxrow - tgt_rows
        if coming_from == 'above' and target.selectable() and (offset_inset > align_bottom):
            if snap_rows >= offset_inset - align_bottom:
                offset_inset = align_bottom
            elif snap_rows >= offset_inset - align_top:
                offset_inset = align_top
            else:
                offset_inset -= snap_rows
        if coming_from == 'below' and target.selectable() and (offset_inset < align_top):
            if snap_rows >= align_top - offset_inset:
                offset_inset = align_top
            elif snap_rows >= align_bottom - offset_inset:
                offset_inset = align_bottom
            else:
                offset_inset += snap_rows
        if offset_inset >= 0:
            self.offset_rows = offset_inset
            self.inset_fraction = (0, 1)
        else:
            if offset_inset + tgt_rows <= 0:
                raise ListBoxError(f'Invalid offset_inset: {offset_inset}, only {tgt_rows} rows in target!')
            self.offset_rows = 0
            self.inset_fraction = (-offset_inset, tgt_rows)
        if cursor_coords is None:
            if coming_from is None:
                return
            cursor_coords = (self.pref_col,)
        if not hasattr(target, 'move_cursor_to_coords'):
            return
        attempt_rows = []
        if len(cursor_coords) == 1:
            pref_col, = cursor_coords
            if coming_from == 'above':
                attempt_rows = range(0, tgt_rows)
            else:
                if coming_from != 'below':
                    raise ValueError("must specify coming_from ('above' or 'below') if cursor row is not specified")
                attempt_rows = range(tgt_rows, -1, -1)
        else:
            pref_col, pref_row = cursor_coords
            if pref_row < 0 or pref_row >= tgt_rows:
                raise ListBoxError(f'cursor_coords row outside valid range for target. pref_row:{pref_row!r} target_rows:{tgt_rows!r}')
            if coming_from == 'above':
                attempt_rows = range(pref_row, -1, -1)
            elif coming_from == 'below':
                attempt_rows = range(pref_row, tgt_rows)
            else:
                attempt_rows = [pref_row]
        for row in attempt_rows:
            if target.move_cursor_to_coords((maxcol,), pref_col, row):
                break

    def get_focus_offset_inset(self, size: tuple[int, int]) -> tuple[int, int]:
        """Return (offset rows, inset rows) for focus widget."""
        maxcol, _maxrow = size
        focus_widget, _pos = self._body.get_focus()
        focus_rows = focus_widget.rows((maxcol,), True)
        offset_rows = self.offset_rows
        inset_rows = 0
        if offset_rows == 0:
            inum, iden = self.inset_fraction
            if inum < 0 or iden < 0 or inum >= iden:
                raise ListBoxError(f'Invalid inset_fraction: {self.inset_fraction!r}')
            inset_rows = focus_rows * inum // iden
            if inset_rows and inset_rows >= focus_rows:
                raise ListBoxError('urwid inset_fraction error (please report)')
        return (offset_rows, inset_rows)

    def make_cursor_visible(self, size: tuple[int, int]) -> None:
        """Shift the focus widget so that its cursor is visible."""
        maxcol, maxrow = size
        focus_widget, _pos = self._body.get_focus()
        if focus_widget is None:
            return
        if not focus_widget.selectable():
            return
        if not hasattr(focus_widget, 'get_cursor_coords'):
            return
        cursor = focus_widget.get_cursor_coords((maxcol,))
        if cursor is None:
            return
        _cx, cy = cursor
        offset_rows, inset_rows = self.get_focus_offset_inset((maxcol, maxrow))
        if cy < inset_rows:
            self.shift_focus((maxcol, maxrow), -cy)
            return
        if offset_rows - inset_rows + cy >= maxrow:
            self.shift_focus((maxcol, maxrow), maxrow - cy - 1)
            return

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        """Move selection through the list elements scrolling when
        necessary. Keystrokes are first passed to widget in focus
        in case that widget can handle them.

        Keystrokes handled by this widget are:
         'up'        up one line (or widget)
         'down'      down one line (or widget)
         'page up'   move cursor up one listbox length (or widget)
         'page down' move cursor down one listbox length (or widget)
        """
        from urwid.command_map import Command
        maxcol, maxrow = size
        if self.set_focus_pending or self.set_focus_valign_pending:
            self._set_focus_complete((maxcol, maxrow), focus=True)
        focus_widget, _pos = self._body.get_focus()
        if focus_widget is None:
            return key
        if focus_widget.selectable():
            key = focus_widget.keypress((maxcol,), key)
            if key is None:
                self.make_cursor_visible((maxcol, maxrow))
                return None

        def actual_key(unhandled) -> str | None:
            if unhandled:
                return key
            return None
        if self._command_map[key] == Command.UP:
            return actual_key(self._keypress_up((maxcol, maxrow)))
        if self._command_map[key] == Command.DOWN:
            return actual_key(self._keypress_down((maxcol, maxrow)))
        if self._command_map[key] == Command.PAGE_UP:
            return actual_key(self._keypress_page_up((maxcol, maxrow)))
        if self._command_map[key] == Command.PAGE_DOWN:
            return actual_key(self._keypress_page_down((maxcol, maxrow)))
        if self._command_map[key] == Command.MAX_LEFT:
            return actual_key(self._keypress_max_left((maxcol, maxrow)))
        if self._command_map[key] == Command.MAX_RIGHT:
            return actual_key(self._keypress_max_right((maxcol, maxrow)))
        return key

    def _keypress_max_left(self, size: tuple[int, int]) -> None:
        self.focus_position = next(iter(self.body.positions()))
        self.set_focus_valign(VAlign.TOP)

    def _keypress_max_right(self, size: tuple[int, int]) -> None:
        self.focus_position = next(iter(self.body.positions(reverse=True)))
        self.set_focus_valign(VAlign.BOTTOM)

    def _keypress_up(self, size: tuple[int, int]) -> bool | None:
        maxcol, maxrow = size
        middle, top, _bottom = self.calculate_visible((maxcol, maxrow), True)
        if middle is None:
            return True
        focus_row_offset, focus_widget, focus_pos, _ignore, cursor = middle
        _trim_top, fill_above = top
        row_offset = focus_row_offset
        pos = focus_pos
        widget = None
        for widget, pos, rows in fill_above:
            row_offset -= rows
            if rows and widget.selectable():
                self.change_focus((maxcol, maxrow), pos, row_offset, 'below')
                return None
        row_offset += 1
        self._invalidate()
        while row_offset > 0:
            widget, pos = self._body.get_prev(pos)
            if widget is None:
                return True
            rows = widget.rows((maxcol,), True)
            row_offset -= rows
            if rows and widget.selectable():
                self.change_focus((maxcol, maxrow), pos, row_offset, 'below')
                return None
        if not focus_widget.selectable() or focus_row_offset + 1 >= maxrow:
            if widget is None:
                self.shift_focus((maxcol, maxrow), row_offset)
                return None
            self.change_focus((maxcol, maxrow), pos, row_offset, 'below')
            return None
        if cursor is not None:
            _x, y = cursor
            if y + focus_row_offset + 1 >= maxrow:
                if widget is None:
                    widget, pos = self._body.get_prev(pos)
                    if widget is None:
                        return None
                    rows = widget.rows((maxcol,), True)
                    row_offset -= rows
                if -row_offset >= rows:
                    row_offset = -(rows - 1)
                self.change_focus((maxcol, maxrow), pos, row_offset, 'below')
                return None
        self.shift_focus((maxcol, maxrow), focus_row_offset + 1)
        return None

    def _keypress_down(self, size: tuple[int, int]) -> bool | None:
        maxcol, maxrow = size
        middle, _top, bottom = self.calculate_visible((maxcol, maxrow), True)
        if middle is None:
            return True
        focus_row_offset, focus_widget, focus_pos, focus_rows, cursor = middle
        _trim_bottom, fill_below = bottom
        row_offset = focus_row_offset + focus_rows
        rows = focus_rows
        pos = focus_pos
        widget = None
        for widget, pos, rows in fill_below:
            if rows and widget.selectable():
                self.change_focus((maxcol, maxrow), pos, row_offset, 'above')
                return None
            row_offset += rows
        row_offset -= 1
        self._invalidate()
        while row_offset < maxrow:
            widget, pos = self._body.get_next(pos)
            if widget is None:
                return True
            rows = widget.rows((maxcol,))
            if rows and widget.selectable():
                self.change_focus((maxcol, maxrow), pos, row_offset, 'above')
                return None
            row_offset += rows
        if not focus_widget.selectable() or focus_row_offset + focus_rows - 1 <= 0:
            if widget is None:
                self.shift_focus((maxcol, maxrow), row_offset - rows)
                return None
            self.change_focus((maxcol, maxrow), pos, row_offset - rows, 'above')
            return None
        if cursor is not None:
            _x, y = cursor
            if y + focus_row_offset - 1 < 0:
                if widget is None:
                    widget, pos = self._body.get_next(pos)
                    if widget is None:
                        return None
                else:
                    row_offset -= rows
                if row_offset >= maxrow:
                    row_offset = maxrow - 1
                self.change_focus((maxcol, maxrow), pos, row_offset, 'above')
                return None
        self.shift_focus((maxcol, maxrow), focus_row_offset - 1)
        return None

    def _keypress_page_up(self, size: tuple[int, int]) -> bool | None:
        maxcol, maxrow = size
        middle, top, _bottom = self.calculate_visible((maxcol, maxrow), True)
        if middle is None:
            return True
        row_offset, focus_widget, focus_pos, focus_rows, cursor = middle
        _trim_top, fill_above = top
        topmost_visible = row_offset
        if not focus_widget.selectable():
            scroll_from_row = topmost_visible
        elif cursor is not None:
            _x, y = cursor
            scroll_from_row = -y
        elif row_offset >= 0:
            scroll_from_row = 0
        else:
            scroll_from_row = topmost_visible
        snap_rows = topmost_visible - scroll_from_row
        row_offset = scroll_from_row + maxrow
        scroll_from_row = topmost_visible = None
        t = [(row_offset, focus_widget, focus_pos, focus_rows)]
        pos = focus_pos
        for widget, pos, rows in fill_above:
            row_offset -= rows
            t.append((row_offset, widget, pos, rows))
        snap_region_start = len(t)
        while row_offset > -snap_rows:
            widget, pos = self._body.get_prev(pos)
            if widget is None:
                break
            rows = widget.rows((maxcol,))
            row_offset -= rows
            if row_offset > 0:
                snap_region_start += 1
            t.append((row_offset, widget, pos, rows))
        row_offset, _w, _p, _r = t[-1]
        if row_offset > 0:
            adjust = -row_offset
            t = [(ro + adjust, w, p, r) for ro, w, p, r in t]
        row_offset, _w, _p, _r = t[0]
        if row_offset >= maxrow:
            del t[0]
            snap_region_start -= 1
        self.update_pref_col_from_focus((maxcol, maxrow))
        search_order = list(range(snap_region_start, len(t))) + list(range(snap_region_start - 1, -1, -1))
        bad_choices = []
        cut_off_selectable_chosen = 0
        for i in search_order:
            row_offset, widget, pos, rows = t[i]
            if not widget.selectable():
                continue
            if not rows:
                continue
            pref_row = max(0, -row_offset)
            if rows + row_offset <= 0:
                self.change_focus((maxcol, maxrow), pos, -(rows - 1), 'below', (self.pref_col, rows - 1), snap_rows - (-row_offset - (rows - 1)))
            else:
                self.change_focus((maxcol, maxrow), pos, row_offset, 'below', (self.pref_col, pref_row), snap_rows)
            if fill_above and self._body.get_prev(fill_above[-1][1]) == (None, None):
                pass
            middle, top, _bottom = self.calculate_visible((maxcol, maxrow), True)
            act_row_offset, _ign1, _ign2, _ign3, _ign4 = middle
            if act_row_offset > row_offset + snap_rows:
                bad_choices.append(i)
                continue
            if act_row_offset < row_offset:
                bad_choices.append(i)
                continue
            if act_row_offset < 0:
                bad_choices.append(i)
                cut_off_selectable_chosen = 1
                continue
            return None
        if cut_off_selectable_chosen:
            return None
        if fill_above and focus_widget.selectable() and (self._body.get_prev(fill_above[-1][1]) == (None, None)):
            pass
        good_choices = [j for j in search_order if j not in bad_choices]
        for i in good_choices + search_order:
            row_offset, widget, pos, rows = t[i]
            if pos == focus_pos:
                continue
            if not rows:
                continue
            if rows + row_offset <= 0:
                snap_rows -= -row_offset - (rows - 1)
                row_offset = -(rows - 1)
            self.change_focus((maxcol, maxrow), pos, row_offset, 'below', None, snap_rows)
            return None
        self.shift_focus((maxcol, maxrow), min(maxrow - 1, row_offset))
        middle, top, _bottom = self.calculate_visible((maxcol, maxrow), True)
        act_row_offset, _ign1, pos, _ign2, _ign3 = middle
        if act_row_offset >= row_offset:
            return None
        if not t:
            return None
        _ign1, _ign2, pos, _ign3 = t[-1]
        widget, pos = self._body.get_prev(pos)
        if widget is None:
            return None
        rows = widget.rows((maxcol,), True)
        self.change_focus((maxcol, maxrow), pos, -(rows - 1), 'below', (self.pref_col, rows - 1), 0)
        return None

    def _keypress_page_down(self, size: tuple[int, int]) -> bool | None:
        maxcol, maxrow = size
        middle, _top, bottom = self.calculate_visible((maxcol, maxrow), True)
        if middle is None:
            return True
        row_offset, focus_widget, focus_pos, focus_rows, cursor = middle
        _trim_bottom, fill_below = bottom
        bottom_edge = maxrow - row_offset
        if not focus_widget.selectable():
            scroll_from_row = bottom_edge
        elif cursor is not None:
            _x, y = cursor
            scroll_from_row = y + 1
        elif bottom_edge >= focus_rows:
            scroll_from_row = focus_rows
        else:
            scroll_from_row = bottom_edge
        snap_rows = bottom_edge - scroll_from_row
        row_offset = -scroll_from_row
        scroll_from_row = bottom_edge = None
        t = [(row_offset, focus_widget, focus_pos, focus_rows)]
        pos = focus_pos
        row_offset += focus_rows
        for widget, pos, rows in fill_below:
            t.append((row_offset, widget, pos, rows))
            row_offset += rows
        snap_region_start = len(t)
        while row_offset < maxrow + snap_rows:
            widget, pos = self._body.get_next(pos)
            if widget is None:
                break
            rows = widget.rows((maxcol,))
            t.append((row_offset, widget, pos, rows))
            row_offset += rows
            if row_offset < maxrow:
                snap_region_start += 1
        row_offset, _w, _p, rows = t[-1]
        if row_offset + rows < maxrow:
            adjust = maxrow - (row_offset + rows)
            t = [(ro + adjust, w, p, r) for ro, w, p, r in t]
        row_offset, _w, _p, rows = t[0]
        if row_offset + rows <= 0:
            del t[0]
            snap_region_start -= 1
        self.update_pref_col_from_focus((maxcol, maxrow))
        search_order = list(range(snap_region_start, len(t))) + list(range(snap_region_start - 1, -1, -1))
        bad_choices = []
        cut_off_selectable_chosen = 0
        for i in search_order:
            row_offset, widget, pos, rows = t[i]
            if not widget.selectable():
                continue
            if not rows:
                continue
            pref_row = min(maxrow - row_offset - 1, rows - 1)
            if row_offset >= maxrow:
                self.change_focus((maxcol, maxrow), pos, maxrow - 1, 'above', (self.pref_col, 0), snap_rows + maxrow - row_offset - 1)
            else:
                self.change_focus((maxcol, maxrow), pos, row_offset, 'above', (self.pref_col, pref_row), snap_rows)
            middle, _top, bottom = self.calculate_visible((maxcol, maxrow), True)
            act_row_offset, _ign1, _ign2, _ign3, _ign4 = middle
            if act_row_offset < row_offset - snap_rows:
                bad_choices.append(i)
                continue
            if act_row_offset > row_offset:
                bad_choices.append(i)
                continue
            if act_row_offset + rows > maxrow:
                bad_choices.append(i)
                cut_off_selectable_chosen = 1
                continue
            return None
        if cut_off_selectable_chosen:
            return None
        good_choices = [j for j in search_order if j not in bad_choices]
        for i in good_choices + search_order:
            row_offset, widget, pos, rows = t[i]
            if pos == focus_pos:
                continue
            if not rows:
                continue
            if row_offset >= maxrow:
                snap_rows -= snap_rows + maxrow - row_offset - 1
                row_offset = maxrow - 1
            self.change_focus((maxcol, maxrow), pos, row_offset, 'above', None, snap_rows)
            return None
        self.shift_focus((maxcol, maxrow), max(1 - focus_rows, row_offset))
        middle, _top, bottom = self.calculate_visible((maxcol, maxrow), True)
        act_row_offset, _ign1, pos, _ign2, _ign3 = middle
        if act_row_offset <= row_offset:
            return None
        if not t:
            return None
        _ign1, _ign2, pos, _ign3 = t[-1]
        widget, pos = self._body.get_next(pos)
        if widget is None:
            return None
        rows = widget.rows((maxcol,), True)
        self.change_focus((maxcol, maxrow), pos, maxrow - 1, 'above', (self.pref_col, 0), 0)
        return None

    def mouse_event(self, size: tuple[int, int], event, button: int, col: int, row: int, focus: bool) -> bool | None:
        """
        Pass the event to the contained widgets.
        May change focus on button 1 press.
        """
        from urwid.util import is_mouse_press
        maxcol, maxrow = size
        middle, top, bottom = self.calculate_visible((maxcol, maxrow), focus=True)
        if middle is None:
            return False
        _ignore, focus_widget, focus_pos, focus_rows, _cursor = middle
        trim_top, fill_above = top
        _ignore, fill_below = bottom
        fill_above.reverse()
        w_list = [*fill_above, (focus_widget, focus_pos, focus_rows), *fill_below]
        wrow = -trim_top
        for w, w_pos, w_rows in w_list:
            if wrow + w_rows > row:
                break
            wrow += w_rows
        else:
            return False
        focus = focus and w == focus_widget
        if is_mouse_press(event) and button == 1 and w.selectable():
            self.change_focus((maxcol, maxrow), w_pos, wrow)
        if not hasattr(w, 'mouse_event'):
            warnings.warn(f'{w.__class__.__module__}.{w.__class__.__name__} is not subclass of Widget', DeprecationWarning, stacklevel=2)
            return False
        handled = w.mouse_event((maxcol,), event, button, col, row - wrow, focus)
        if handled:
            return True
        if is_mouse_press(event):
            if button == 4:
                return not self._keypress_up((maxcol, maxrow))
            if button == 5:
                return not self._keypress_down((maxcol, maxrow))
        return False

    def ends_visible(self, size: tuple[int, int], focus: bool=False) -> list[Literal['top', 'bottom']]:
        """
        Return a list that may contain ``'top'`` and/or ``'bottom'``.

        i.e. this function will return one of: [], [``'top'``],
        [``'bottom'``] or [``'top'``, ``'bottom'``].

        convenience function for checking whether the top and bottom
        of the list are visible
        """
        maxcol, maxrow = size
        result = []
        middle, top, bottom = self.calculate_visible((maxcol, maxrow), focus=focus)
        if middle is None:
            return ['top', 'bottom']
        trim_top, above = top
        trim_bottom, below = bottom
        if trim_bottom == 0:
            row_offset, _w, pos, rows, _c = middle
            row_offset += rows
            for _w, pos, rows in below:
                row_offset += rows
            if row_offset < maxrow or self._body.get_next(pos) == (None, None):
                result.append('bottom')
        if trim_top == 0:
            row_offset, _w, pos, _rows, _c = middle
            for _w, pos, rows in above:
                row_offset -= rows
            if self._body.get_prev(pos) == (None, None):
                result.insert(0, 'top')
        return result

    def __iter__(self):
        """
        Return an iterator over the positions in this ListBox.

        If self._body does not implement positions() then iterate
        from the focus widget down to the bottom, then from above
        the focus up to the top.  This is the best we can do with
        a minimal list walker implementation.
        """
        positions_fn = getattr(self._body, 'positions', None)
        if positions_fn:
            yield from positions_fn()
            return
        focus_widget, focus_pos = self._body.get_focus()
        if focus_widget is None:
            return
        pos = focus_pos
        while True:
            yield pos
            w, pos = self._body.get_next(pos)
            if not w:
                break
        pos = focus_pos
        while True:
            w, pos = self._body.get_prev(pos)
            if not w:
                break
            yield pos

    def __reversed__(self):
        """
        Return a reversed iterator over the positions in this ListBox.

        If :attr:`body` does not implement :meth:`positions` then iterate
        from above the focus widget up to the top, then from the focus
        widget down to the bottom.  Note that this is not actually the
        reverse of what `__iter__()` produces, but this is the best we can
        do with a minimal list walker implementation.
        """
        positions_fn = getattr(self._body, 'positions', None)
        if positions_fn:
            yield from positions_fn(reverse=True)
            return
        focus_widget, focus_pos = self._body.get_focus()
        if focus_widget is None:
            return
        pos = focus_pos
        while True:
            w, pos = self._body.get_prev(pos)
            if not w:
                break
            yield pos
        pos = focus_pos
        while True:
            yield pos
            w, pos = self._body.get_next(pos)
            if not w:
                break