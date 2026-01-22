from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerErrorbar(HandlerLine2D):
    """Handler for Errorbars."""

    def __init__(self, xerr_size=0.5, yerr_size=None, marker_pad=0.3, numpoints=None, **kwargs):
        self._xerr_size = xerr_size
        self._yerr_size = yerr_size
        super().__init__(marker_pad=marker_pad, numpoints=numpoints, **kwargs)

    def get_err_size(self, legend, xdescent, ydescent, width, height, fontsize):
        xerr_size = self._xerr_size * fontsize
        if self._yerr_size is None:
            yerr_size = xerr_size
        else:
            yerr_size = self._yerr_size * fontsize
        return (xerr_size, yerr_size)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        plotlines, caplines, barlinecols = orig_handle
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        ydata = np.full_like(xdata, (height - ydescent) / 2)
        legline = Line2D(xdata, ydata)
        xdata_marker = np.asarray(xdata_marker)
        ydata_marker = np.asarray(ydata[:len(xdata_marker)])
        xerr_size, yerr_size = self.get_err_size(legend, xdescent, ydescent, width, height, fontsize)
        legline_marker = Line2D(xdata_marker, ydata_marker)
        if plotlines is None:
            legline.set_visible(False)
            legline_marker.set_visible(False)
        else:
            self.update_prop(legline, plotlines, legend)
            legline.set_drawstyle('default')
            legline.set_marker('none')
            self.update_prop(legline_marker, plotlines, legend)
            legline_marker.set_linestyle('None')
            if legend.markerscale != 1:
                newsz = legline_marker.get_markersize() * legend.markerscale
                legline_marker.set_markersize(newsz)
        handle_barlinecols = []
        handle_caplines = []
        if orig_handle.has_xerr:
            verts = [((x - xerr_size, y), (x + xerr_size, y)) for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols.append(coll)
            if caplines:
                capline_left = Line2D(xdata_marker - xerr_size, ydata_marker)
                capline_right = Line2D(xdata_marker + xerr_size, ydata_marker)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker('|')
                capline_right.set_marker('|')
                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)
        if orig_handle.has_yerr:
            verts = [((x, y - yerr_size), (x, y + yerr_size)) for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols.append(coll)
            if caplines:
                capline_left = Line2D(xdata_marker, ydata_marker - yerr_size)
                capline_right = Line2D(xdata_marker, ydata_marker + yerr_size)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker('_')
                capline_right.set_marker('_')
                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)
        artists = [*handle_barlinecols, *handle_caplines, legline, legline_marker]
        for artist in artists:
            artist.set_transform(trans)
        return artists