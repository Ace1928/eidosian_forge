from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
class NavigationToolbar2:
    """
    Base class for the navigation cursor, version 2.

    Backends must implement a canvas that handles connections for
    'button_press_event' and 'button_release_event'.  See
    :meth:`FigureCanvasBase.mpl_connect` for more information.

    They must also define

    :meth:`save_figure`
        Save the current figure.

    :meth:`draw_rubberband` (optional)
        Draw the zoom to rect "rubberband" rectangle.

    :meth:`set_message` (optional)
        Display message.

    :meth:`set_history_buttons` (optional)
        You can change the history back / forward buttons to indicate disabled / enabled
        state.

    and override ``__init__`` to set up the toolbar -- without forgetting to
    call the base-class init.  Typically, ``__init__`` needs to set up toolbar
    buttons connected to the `home`, `back`, `forward`, `pan`, `zoom`, and
    `save_figure` methods and using standard icons in the "images" subdirectory
    of the data path.

    That's it, we'll do the rest!
    """
    toolitems = (('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous view', 'back', 'back'), ('Forward', 'Forward to next view', 'forward', 'forward'), (None, None, None, None), ('Pan', 'Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect', 'move', 'pan'), ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'), ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'), (None, None, None, None), ('Save', 'Save the figure', 'filesave', 'save_figure'))

    def __init__(self, canvas):
        self.canvas = canvas
        canvas.toolbar = self
        self._nav_stack = cbook._Stack()
        self._last_cursor = tools.Cursors.POINTER
        self._id_press = self.canvas.mpl_connect('button_press_event', self._zoom_pan_handler)
        self._id_release = self.canvas.mpl_connect('button_release_event', self._zoom_pan_handler)
        self._id_drag = self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self._pan_info = None
        self._zoom_info = None
        self.mode = _Mode.NONE
        self.set_history_buttons()

    def set_message(self, s):
        """Display a message on toolbar or in status bar."""

    def draw_rubberband(self, event, x0, y0, x1, y1):
        """
        Draw a rectangle rubberband to indicate zoom limits.

        Note that it is not guaranteed that ``x0 <= x1`` and ``y0 <= y1``.
        """

    def remove_rubberband(self):
        """Remove the rubberband."""

    def home(self, *args):
        """
        Restore the original view.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        self._nav_stack.home()
        self.set_history_buttons()
        self._update_view()

    def back(self, *args):
        """
        Move back up the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        self._nav_stack.back()
        self.set_history_buttons()
        self._update_view()

    def forward(self, *args):
        """
        Move forward in the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        self._nav_stack.forward()
        self.set_history_buttons()
        self._update_view()

    def _update_cursor(self, event):
        """
        Update the cursor after a mouse move event or a tool (de)activation.
        """
        if self.mode and event.inaxes and event.inaxes.get_navigate():
            if self.mode == _Mode.ZOOM and self._last_cursor != tools.Cursors.SELECT_REGION:
                self.canvas.set_cursor(tools.Cursors.SELECT_REGION)
                self._last_cursor = tools.Cursors.SELECT_REGION
            elif self.mode == _Mode.PAN and self._last_cursor != tools.Cursors.MOVE:
                self.canvas.set_cursor(tools.Cursors.MOVE)
                self._last_cursor = tools.Cursors.MOVE
        elif self._last_cursor != tools.Cursors.POINTER:
            self.canvas.set_cursor(tools.Cursors.POINTER)
            self._last_cursor = tools.Cursors.POINTER

    @contextmanager
    def _wait_cursor_for_draw_cm(self):
        """
        Set the cursor to a wait cursor when drawing the canvas.

        In order to avoid constantly changing the cursor when the canvas
        changes frequently, do nothing if this context was triggered during the
        last second.  (Optimally we'd prefer only setting the wait cursor if
        the *current* draw takes too long, but the current draw blocks the GUI
        thread).
        """
        self._draw_time, last_draw_time = (time.time(), getattr(self, '_draw_time', -np.inf))
        if self._draw_time - last_draw_time > 1:
            try:
                self.canvas.set_cursor(tools.Cursors.WAIT)
                yield
            finally:
                self.canvas.set_cursor(self._last_cursor)
        else:
            yield

    @staticmethod
    def _mouse_event_to_message(event):
        if event.inaxes and event.inaxes.get_navigate():
            try:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
            except (ValueError, OverflowError):
                pass
            else:
                s = s.rstrip()
                artists = [a for a in event.inaxes._mouseover_set if a.contains(event)[0] and a.get_visible()]
                if artists:
                    a = cbook._topmost_artist(artists)
                    if a is not event.inaxes.patch:
                        data = a.get_cursor_data(event)
                        if data is not None:
                            data_str = a.format_cursor_data(data).rstrip()
                            if data_str:
                                s = s + '\n' + data_str
                return s
        return ''

    def mouse_move(self, event):
        self._update_cursor(event)
        self.set_message(self._mouse_event_to_message(event))

    def _zoom_pan_handler(self, event):
        if self.mode == _Mode.PAN:
            if event.name == 'button_press_event':
                self.press_pan(event)
            elif event.name == 'button_release_event':
                self.release_pan(event)
        if self.mode == _Mode.ZOOM:
            if event.name == 'button_press_event':
                self.press_zoom(event)
            elif event.name == 'button_release_event':
                self.release_zoom(event)

    def pan(self, *args):
        """
        Toggle the pan/zoom tool.

        Pan with left button, zoom with right.
        """
        if not self.canvas.widgetlock.available(self):
            self.set_message('pan unavailable')
            return
        if self.mode == _Mode.PAN:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.PAN
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)
    _PanInfo = namedtuple('_PanInfo', 'button axes cid')

    def press_pan(self, event):
        """Callback for mouse button press in pan/zoom mode."""
        if event.button not in [MouseButton.LEFT, MouseButton.RIGHT] or event.x is None or event.y is None:
            return
        axes = [a for a in self.canvas.figure.get_axes() if a.in_axes(event) and a.get_navigate() and a.can_pan()]
        if not axes:
            return
        if self._nav_stack() is None:
            self.push_current()
        for ax in axes:
            ax.start_pan(event.x, event.y, event.button)
        self.canvas.mpl_disconnect(self._id_drag)
        id_drag = self.canvas.mpl_connect('motion_notify_event', self.drag_pan)
        self._pan_info = self._PanInfo(button=event.button, axes=axes, cid=id_drag)

    def drag_pan(self, event):
        """Callback for dragging in pan/zoom mode."""
        for ax in self._pan_info.axes:
            ax.drag_pan(self._pan_info.button, event.key, event.x, event.y)
        self.canvas.draw_idle()

    def release_pan(self, event):
        """Callback for mouse button release in pan/zoom mode."""
        if self._pan_info is None:
            return
        self.canvas.mpl_disconnect(self._pan_info.cid)
        self._id_drag = self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        for ax in self._pan_info.axes:
            ax.end_pan()
        self.canvas.draw_idle()
        self._pan_info = None
        self.push_current()

    def zoom(self, *args):
        if not self.canvas.widgetlock.available(self):
            self.set_message('zoom unavailable')
            return
        'Toggle zoom to rect mode.'
        if self.mode == _Mode.ZOOM:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.ZOOM
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)
    _ZoomInfo = namedtuple('_ZoomInfo', 'direction start_xy axes cid cbar')

    def press_zoom(self, event):
        """Callback for mouse button press in zoom to rect mode."""
        if event.button not in [MouseButton.LEFT, MouseButton.RIGHT] or event.x is None or event.y is None:
            return
        axes = [a for a in self.canvas.figure.get_axes() if a.in_axes(event) and a.get_navigate() and a.can_zoom()]
        if not axes:
            return
        if self._nav_stack() is None:
            self.push_current()
        id_zoom = self.canvas.mpl_connect('motion_notify_event', self.drag_zoom)
        if hasattr(axes[0], '_colorbar'):
            cbar = axes[0]._colorbar.orientation
        else:
            cbar = None
        self._zoom_info = self._ZoomInfo(direction='in' if event.button == 1 else 'out', start_xy=(event.x, event.y), axes=axes, cid=id_zoom, cbar=cbar)

    def drag_zoom(self, event):
        """Callback for dragging in zoom mode."""
        start_xy = self._zoom_info.start_xy
        ax = self._zoom_info.axes[0]
        (x1, y1), (x2, y2) = np.clip([start_xy, [event.x, event.y]], ax.bbox.min, ax.bbox.max)
        key = event.key
        if self._zoom_info.cbar == 'horizontal':
            key = 'x'
        elif self._zoom_info.cbar == 'vertical':
            key = 'y'
        if key == 'x':
            y1, y2 = ax.bbox.intervaly
        elif key == 'y':
            x1, x2 = ax.bbox.intervalx
        self.draw_rubberband(event, x1, y1, x2, y2)

    def release_zoom(self, event):
        """Callback for mouse button release in zoom to rect mode."""
        if self._zoom_info is None:
            return
        self.canvas.mpl_disconnect(self._zoom_info.cid)
        self.remove_rubberband()
        start_x, start_y = self._zoom_info.start_xy
        key = event.key
        if self._zoom_info.cbar == 'horizontal':
            key = 'x'
        elif self._zoom_info.cbar == 'vertical':
            key = 'y'
        if abs(event.x - start_x) < 5 and key != 'y' or (abs(event.y - start_y) < 5 and key != 'x'):
            self.canvas.draw_idle()
            self._zoom_info = None
            return
        for i, ax in enumerate(self._zoom_info.axes):
            twinx = any((ax.get_shared_x_axes().joined(ax, prev) for prev in self._zoom_info.axes[:i]))
            twiny = any((ax.get_shared_y_axes().joined(ax, prev) for prev in self._zoom_info.axes[:i]))
            ax._set_view_from_bbox((start_x, start_y, event.x, event.y), self._zoom_info.direction, key, twinx, twiny)
        self.canvas.draw_idle()
        self._zoom_info = None
        self.push_current()

    def push_current(self):
        """Push the current view limits and position onto the stack."""
        self._nav_stack.push(WeakKeyDictionary({ax: (ax._get_view(), (ax.get_position(True).frozen(), ax.get_position().frozen())) for ax in self.canvas.figure.axes}))
        self.set_history_buttons()

    def _update_view(self):
        """
        Update the viewlim and position from the view and position stack for
        each Axes.
        """
        nav_info = self._nav_stack()
        if nav_info is None:
            return
        items = list(nav_info.items())
        for ax, (view, (pos_orig, pos_active)) in items:
            ax._set_view(view)
            ax._set_position(pos_orig, 'original')
            ax._set_position(pos_active, 'active')
        self.canvas.draw_idle()

    def configure_subplots(self, *args):
        if hasattr(self, 'subplot_tool'):
            self.subplot_tool.figure.canvas.manager.show()
            return
        from matplotlib.figure import Figure
        with mpl.rc_context({'toolbar': 'none'}):
            manager = type(self.canvas).new_manager(Figure(figsize=(6, 3)), -1)
        manager.set_window_title('Subplot configuration tool')
        tool_fig = manager.canvas.figure
        tool_fig.subplots_adjust(top=0.9)
        self.subplot_tool = widgets.SubplotTool(self.canvas.figure, tool_fig)
        cid = self.canvas.mpl_connect('close_event', lambda e: manager.destroy())

        def on_tool_fig_close(e):
            self.canvas.mpl_disconnect(cid)
            del self.subplot_tool
        tool_fig.canvas.mpl_connect('close_event', on_tool_fig_close)
        manager.show()
        return self.subplot_tool

    def save_figure(self, *args):
        """Save the current figure."""
        raise NotImplementedError

    def update(self):
        """Reset the Axes stack."""
        self._nav_stack.clear()
        self.set_history_buttons()

    def set_history_buttons(self):
        """Enable or disable the back/forward button."""