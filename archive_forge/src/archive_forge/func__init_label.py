from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
def _init_label(self, **kwargs):
    tr = self._axis_artist_helper.get_axislabel_transform(self.axes) + self.offset_transform
    self.label = AxisLabel(0, 0, '__from_axes__', color='auto', fontsize=kwargs.get('labelsize', mpl.rcParams['axes.labelsize']), fontweight=mpl.rcParams['axes.labelweight'], axis=self.axis, transform=tr, axis_direction=self._axis_direction)
    self.label.set_figure(self.axes.figure)
    labelpad = kwargs.get('labelpad', 5)
    self.label.set_pad(labelpad)