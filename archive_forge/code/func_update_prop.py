from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
def update_prop(self, legend_handle, orig_handle, legend):
    self._update_prop(legend_handle, orig_handle)
    legend_handle.set_figure(legend.figure)
    legend_handle.set_clip_box(None)
    legend_handle.set_clip_path(None)