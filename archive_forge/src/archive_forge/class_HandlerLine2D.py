from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerLine2D(HandlerNpoints):
    """
    Handler for `.Line2D` instances.

    See Also
    --------
    HandlerLine2DCompound : An earlier handler implementation, which used one
                            artist for the line and another for the marker(s).
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        markevery = None
        if self.get_numpoints(legend) == 1:
            xdata = np.linspace(xdata[0], xdata[-1], 3)
            markevery = [1]
        ydata = np.full_like(xdata, (height - ydescent) / 2)
        legline = Line2D(xdata, ydata, markevery=markevery)
        self.update_prop(legline, orig_handle, legend)
        if legend.markerscale != 1:
            newsz = legline.get_markersize() * legend.markerscale
            legline.set_markersize(newsz)
        legline.set_transform(trans)
        return [legline]