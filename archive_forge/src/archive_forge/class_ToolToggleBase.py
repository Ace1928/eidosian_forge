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
class ToolToggleBase(ToolBase):
    """
    Toggleable tool.

    Every time it is triggered, it switches between enable and disable.

    Parameters
    ----------
    ``*args``
        Variable length argument to be used by the Tool.
    ``**kwargs``
        `toggled` if present and True, sets the initial state of the Tool
        Arbitrary keyword arguments to be consumed by the Tool
    """
    radio_group = None
    "\n    Attribute to group 'radio' like tools (mutually exclusive).\n\n    `str` that identifies the group or **None** if not belonging to a group.\n    "
    cursor = None
    'Cursor to use when the tool is active.'
    default_toggled = False
    'Default of toggled state.'

    def __init__(self, *args, **kwargs):
        self._toggled = kwargs.pop('toggled', self.default_toggled)
        super().__init__(*args, **kwargs)

    def trigger(self, sender, event, data=None):
        """Calls `enable` or `disable` based on `toggled` value."""
        if self._toggled:
            self.disable(event)
        else:
            self.enable(event)
        self._toggled = not self._toggled

    def enable(self, event=None):
        """
        Enable the toggle tool.

        `trigger` calls this method when `toggled` is False.
        """
        pass

    def disable(self, event=None):
        """
        Disable the toggle tool.

        `trigger` call this method when `toggled` is True.

        This can happen in different circumstances.

        * Click on the toolbar tool button.
        * Call to `matplotlib.backend_managers.ToolManager.trigger_tool`.
        * Another `ToolToggleBase` derived tool is triggered
          (from the same `.ToolManager`).
        """
        pass

    @property
    def toggled(self):
        """State of the toggled tool."""
        return self._toggled

    def set_figure(self, figure):
        toggled = self.toggled
        if toggled:
            if self.figure:
                self.trigger(self, None)
            else:
                self._toggled = False
        super().set_figure(figure)
        if toggled:
            if figure:
                self.trigger(self, None)
            else:
                self._toggled = True