from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
def adjust_drawing_area(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    xdescent = xdescent - self._xpad * fontsize
    ydescent = ydescent - self._ypad * fontsize
    width = width - self._xpad * fontsize
    height = height - self._ypad * fontsize
    return (xdescent, ydescent, width, height)