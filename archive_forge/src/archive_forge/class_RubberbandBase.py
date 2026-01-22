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
class RubberbandBase(ToolBase):
    """Draw and remove a rubberband."""

    def trigger(self, sender, event, data=None):
        """Call `draw_rubberband` or `remove_rubberband` based on data."""
        if not self.figure.canvas.widgetlock.available(sender):
            return
        if data is not None:
            self.draw_rubberband(*data)
        else:
            self.remove_rubberband()

    def draw_rubberband(self, *data):
        """
        Draw rubberband.

        This method must get implemented per backend.
        """
        raise NotImplementedError

    def remove_rubberband(self):
        """
        Remove rubberband.

        This method should get implemented per backend.
        """
        pass