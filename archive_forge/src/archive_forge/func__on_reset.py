from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def _on_reset(self, event):
    with ExitStack() as stack:
        stack.enter_context(cbook._setattr_cm(self, drawon=False))
        for slider in self._sliders:
            stack.enter_context(cbook._setattr_cm(slider, drawon=False, eventson=False))
        for slider in self._sliders:
            slider.reset()
    if self.drawon:
        event.canvas.draw()
    self._on_slider_changed(None)