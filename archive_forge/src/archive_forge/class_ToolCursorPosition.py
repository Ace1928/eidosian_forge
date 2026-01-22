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
class ToolCursorPosition(ToolBase):
    """
    Send message with the current pointer position.

    This tool runs in the background reporting the position of the cursor.
    """

    def __init__(self, *args, **kwargs):
        self._id_drag = None
        super().__init__(*args, **kwargs)

    def set_figure(self, figure):
        if self._id_drag:
            self.canvas.mpl_disconnect(self._id_drag)
        super().set_figure(figure)
        if figure:
            self._id_drag = self.canvas.mpl_connect('motion_notify_event', self.send_message)

    def send_message(self, event):
        """Call `matplotlib.backend_managers.ToolManager.message_event`."""
        if self.toolmanager.messagelock.locked():
            return
        from matplotlib.backend_bases import NavigationToolbar2
        message = NavigationToolbar2._mouse_event_to_message(event)
        self.toolmanager.message_event(message, self)