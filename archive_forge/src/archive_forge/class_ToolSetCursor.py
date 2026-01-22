import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
class ToolSetCursor(ToolBase):
    """
    Change to the current cursor while inaxes.

    This tool, keeps track of all `ToolToggleBase` derived tools, and updates
    the cursor when a tool gets triggered.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id_drag = None
        self._current_tool = None
        self._default_cursor = cursors.POINTER
        self._last_cursor = self._default_cursor
        self.toolmanager.toolmanager_connect('tool_added_event', self._add_tool_cbk)
        for tool in self.toolmanager.tools.values():
            self._add_tool(tool)

    def set_figure(self, figure):
        if self._id_drag:
            self.canvas.mpl_disconnect(self._id_drag)
        super().set_figure(figure)
        if figure:
            self._id_drag = self.canvas.mpl_connect('motion_notify_event', self._set_cursor_cbk)

    def _tool_trigger_cbk(self, event):
        if event.tool.toggled:
            self._current_tool = event.tool
        else:
            self._current_tool = None
        self._set_cursor_cbk(event.canvasevent)

    def _add_tool(self, tool):
        """Set the cursor when the tool is triggered."""
        if getattr(tool, 'cursor', None) is not None:
            self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name, self._tool_trigger_cbk)

    def _add_tool_cbk(self, event):
        """Process every newly added tool."""
        if event.tool is self:
            return
        self._add_tool(event.tool)

    def _set_cursor_cbk(self, event):
        if not event or not self.canvas:
            return
        if self._current_tool and getattr(event, 'inaxes', None) and event.inaxes.get_navigate():
            if self._last_cursor != self._current_tool.cursor:
                self.canvas.set_cursor(self._current_tool.cursor)
                self._last_cursor = self._current_tool.cursor
        elif self._last_cursor != self._default_cursor:
            self.canvas.set_cursor(self._default_cursor)
            self._last_cursor = self._default_cursor