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
class ToolPan(ZoomPanBase):
    """Pan axes with left mouse, zoom with right."""
    default_keymap = property(lambda self: mpl.rcParams['keymap.pan'])
    description = 'Pan axes with left mouse, zoom with right'
    image = 'move'
    cursor = cursors.MOVE
    radio_group = 'default'

    def __init__(self, *args):
        super().__init__(*args)
        self._id_drag = None

    def _cancel_action(self):
        self._button_pressed = None
        self._xypress = []
        self.figure.canvas.mpl_disconnect(self._id_drag)
        self.toolmanager.messagelock.release(self)
        self.figure.canvas.draw_idle()

    def _press(self, event):
        if event.button == 1:
            self._button_pressed = 1
        elif event.button == 3:
            self._button_pressed = 3
        else:
            self._cancel_action()
            return
        x, y = (event.x, event.y)
        self._xypress = []
        for i, a in enumerate(self.figure.get_axes()):
            if x is not None and y is not None and a.in_axes(event) and a.get_navigate() and a.can_pan():
                a.start_pan(x, y, event.button)
                self._xypress.append((a, i))
                self.toolmanager.messagelock(self)
                self._id_drag = self.figure.canvas.mpl_connect('motion_notify_event', self._mouse_move)

    def _release(self, event):
        if self._button_pressed is None:
            self._cancel_action()
            return
        self.figure.canvas.mpl_disconnect(self._id_drag)
        self.toolmanager.messagelock.release(self)
        for a, _ind in self._xypress:
            a.end_pan()
        if not self._xypress:
            self._cancel_action()
            return
        self.toolmanager.get_tool(_views_positions).push_current()
        self._cancel_action()

    def _mouse_move(self, event):
        for a, _ind in self._xypress:
            a.drag_pan(self._button_pressed, event.key, event.x, event.y)
        self.toolmanager.canvas.draw_idle()