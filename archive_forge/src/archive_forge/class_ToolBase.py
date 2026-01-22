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
class ToolBase:
    """
    Base tool class.

    A base tool, only implements `trigger` method or no method at all.
    The tool is instantiated by `matplotlib.backend_managers.ToolManager`.
    """
    default_keymap = None
    '\n    Keymap to associate with this tool.\n\n    ``list[str]``: List of keys that will trigger this tool when a keypress\n    event is emitted on ``self.figure.canvas``.  Note that this attribute is\n    looked up on the instance, and can therefore be a property (this is used\n    e.g. by the built-in tools to load the rcParams at instantiation time).\n    '
    description = None
    '\n    Description of the Tool.\n\n    `str`: Tooltip used if the Tool is included in a Toolbar.\n    '
    image = None
    '\n    Filename of the image.\n\n    `str`: Filename of the image to use in a Toolbar.  If None, the *name* is\n    used as a label in the toolbar button.\n    '

    def __init__(self, toolmanager, name):
        self._name = name
        self._toolmanager = toolmanager
        self._figure = None
    name = property(lambda self: self._name, doc='The tool id (str, must be unique among tools of a tool manager).')
    toolmanager = property(lambda self: self._toolmanager, doc='The `.ToolManager` that controls this tool.')
    canvas = property(lambda self: self._figure.canvas if self._figure is not None else None, doc='The canvas of the figure affected by this tool, or None.')

    def set_figure(self, figure):
        self._figure = figure
    figure = property(lambda self: self._figure, lambda self, figure: self.set_figure(figure), doc='The Figure affected by this tool, or None.')

    def _make_classic_style_pseudo_toolbar(self):
        """
        Return a placeholder object with a single `canvas` attribute.

        This is useful to reuse the implementations of tools already provided
        by the classic Toolbars.
        """
        return SimpleNamespace(canvas=self.canvas)

    def trigger(self, sender, event, data=None):
        """
        Called when this tool gets used.

        This method is called by `.ToolManager.trigger_tool`.

        Parameters
        ----------
        event : `.Event`
            The canvas event that caused this tool to be called.
        sender : object
            Object that requested the tool to be triggered.
        data : object
            Extra data.
        """
        pass