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
def add_tool(self, tool, group, position=-1):
    """
        Add a tool to this container.

        Parameters
        ----------
        tool : tool_like
            The tool to add, see `.ToolManager.get_tool`.
        group : str
            The name of the group to add this tool to.
        position : int, default: -1
            The position within the group to place this tool.
        """
    tool = self.toolmanager.get_tool(tool)
    image = self._get_image_filename(tool.image)
    toggle = getattr(tool, 'toggled', None) is not None
    self.add_toolitem(tool.name, group, position, image, tool.description, toggle)
    if toggle:
        self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name, self._tool_toggled_cbk)
        if tool.toggled:
            self.toggle_toolitem(tool.name, True)