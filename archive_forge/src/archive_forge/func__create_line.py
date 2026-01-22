from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
@staticmethod
def _create_line(orig_handle, width, height):
    legline = Line2D([0, width], [height / 2, height / 2], color=orig_handle.get_edgecolor(), linestyle=orig_handle.get_linestyle(), linewidth=orig_handle.get_linewidth())
    legline.set_drawstyle('default')
    legline.set_marker('')
    return legline