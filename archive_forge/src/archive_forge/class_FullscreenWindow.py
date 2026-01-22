from typing import (
from types import TracebackType
import logging
import re
import sys
import blessed
from .formatstring import fmtstr, FmtStr
from .formatstringarray import FSArray
from .termhelpers import Cbreak
class FullscreenWindow(BaseWindow, ContextManager['FullscreenWindow']):
    """2D-text rendering window that disappears when its context is left

    FullscreenWindow will only render arrays the size of the terminal
    or smaller, and leaves no trace on exit (like top or vim). It never
    scrolls the terminal. Changing the terminal size doesn't do anything,
    but rendered arrays need to fit on the screen.

    Note:
        The context of the FullscreenWindow
        object must be entered before calling any of its methods.

        Within the context of CursorAwareWindow, refrain from writing to
        its out_stream; cached writes will be inaccurate.
    """

    def __init__(self, out_stream: Optional[IO]=None, hide_cursor: bool=True) -> None:
        """Constructs a FullscreenWindow

        Args:
            out_stream (file): Defaults to sys.__stdout__
            hide_cursor (bool): Hides cursor while in context
        """
        super().__init__(out_stream=out_stream, hide_cursor=hide_cursor)
        self.fullscreen_ctx = self.t.fullscreen()

    def __enter__(self) -> 'FullscreenWindow':
        self.fullscreen_ctx.__enter__()
        return super().__enter__()

    def __exit__(self, type: Optional[Type[BaseException]]=None, value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        self.fullscreen_ctx.__exit__(type, value, traceback)
        super().__exit__(type, value, traceback)

    def render_to_terminal(self, array: Union[FSArray, List[FmtStr]], cursor_pos: Tuple[int, int]=(0, 0)) -> None:
        """Renders array to terminal and places (0-indexed) cursor

        Args:
            array (FSArray): Grid of styled characters to be rendered.

        * If array received is of width too small, render it anyway
        * If array received is of width too large,
        * render the renderable portion
        * If array received is of height too small, render it anyway
        * If array received is of height too large,
        * render the renderable portion (no scroll)
        """
        height, width = (self.height, self.width)
        for_stdout = self.fmtstr_to_stdout_xform()
        if not self.hide_cursor:
            self.write(self.t.hide_cursor)
        if height != self._last_rendered_height or width != self._last_rendered_width:
            self.on_terminal_size_change(height, width)
        current_lines_by_row: Dict[int, Optional[FmtStr]] = {}
        for row, line in enumerate(array):
            current_lines_by_row[row] = line
            if line == self._last_lines_by_row.get(row, None):
                continue
            self.write(self.t.move(row, 0))
            self.write(for_stdout(line))
            if len(line) < width:
                self.write(self.t.clear_eol)
        for row in range(len(array), height):
            if self._last_lines_by_row and row not in self._last_lines_by_row:
                continue
            self.write(self.t.move(row, 0))
            self.write(self.t.clear_eol)
            self.write(self.t.clear_bol)
            current_lines_by_row[row] = None
        logger.debug('lines in last lines by row: %r' % self._last_lines_by_row.keys())
        logger.debug('lines in current lines by row: %r' % current_lines_by_row.keys())
        self.write(self.t.move(*cursor_pos))
        self._last_lines_by_row = current_lines_by_row
        if not self.hide_cursor:
            self.write(self.t.normal_cursor)