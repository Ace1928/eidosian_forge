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
def _on_slider_changed(self, _):
    self.targetfig.subplots_adjust(**{slider.label.get_text(): slider.val for slider in self._sliders})
    if self.drawon:
        self.targetfig.canvas.draw()