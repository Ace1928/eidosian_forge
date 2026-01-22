import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
class _HeatMapper:
    """Draw a heatmap plot of a matrix with nice labels and colormaps."""

    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, cbar, cbar_kws, xticklabels=True, yticklabels=True, mask=None):
        """Initialize the plotting object."""
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)
        mask = _matrix_mask(data, mask)
        plot_data = np.ma.masked_where(np.asarray(mask), plot_data)
        xtickevery = 1
        if isinstance(xticklabels, int):
            xtickevery = xticklabels
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is True:
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is False:
            xticklabels = []
        ytickevery = 1
        if isinstance(yticklabels, int):
            ytickevery = yticklabels
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is True:
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is False:
            yticklabels = []
        if not len(xticklabels):
            self.xticks = []
            self.xticklabels = []
        elif isinstance(xticklabels, str) and xticklabels == 'auto':
            self.xticks = 'auto'
            self.xticklabels = _index_to_ticklabels(data.columns)
        else:
            self.xticks, self.xticklabels = self._skip_ticks(xticklabels, xtickevery)
        if not len(yticklabels):
            self.yticks = []
            self.yticklabels = []
        elif isinstance(yticklabels, str) and yticklabels == 'auto':
            self.yticks = 'auto'
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.yticks, self.yticklabels = self._skip_ticks(yticklabels, ytickevery)
        xlabel = _index_to_label(data.columns)
        ylabel = _index_to_label(data.index)
        self.xlabel = xlabel if xlabel is not None else ''
        self.ylabel = ylabel if ylabel is not None else ''
        self._determine_cmap_params(plot_data, vmin, vmax, cmap, center, robust)
        if annot is None or annot is False:
            annot = False
            annot_data = None
        else:
            if isinstance(annot, bool):
                annot_data = plot_data
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != plot_data.shape:
                    err = '`data` and `annot` must have same shape.'
                    raise ValueError(err)
            annot = True
        self.data = data
        self.plot_data = plot_data
        self.annot = annot
        self.annot_data = annot_data
        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws.copy()
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws.copy()

    def _determine_cmap_params(self, plot_data, vmin, vmax, cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""
        calc_data = plot_data.astype(float).filled(np.nan)
        if vmin is None:
            if robust:
                vmin = np.nanpercentile(calc_data, 2)
            else:
                vmin = np.nanmin(calc_data)
        if vmax is None:
            if robust:
                vmax = np.nanpercentile(calc_data, 98)
            else:
                vmax = np.nanmax(calc_data)
        self.vmin, self.vmax = (vmin, vmax)
        if cmap is None:
            if center is None:
                self.cmap = cm.rocket
            else:
                self.cmap = cm.icefire
        elif isinstance(cmap, str):
            self.cmap = get_colormap(cmap)
        elif isinstance(cmap, list):
            self.cmap = mpl.colors.ListedColormap(cmap)
        else:
            self.cmap = cmap
        if center is not None:
            bad = self.cmap(np.ma.masked_invalid([np.nan]))[0]
            under = self.cmap(-np.inf)
            over = self.cmap(np.inf)
            under_set = under != self.cmap(0)
            over_set = over != self.cmap(self.cmap.N - 1)
            vrange = max(vmax - center, center - vmin)
            normlize = mpl.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            self.cmap = mpl.colors.ListedColormap(self.cmap(cc))
            self.cmap.set_bad(bad)
            if under_set:
                self.cmap.set_under(under)
            if over_set:
                self.cmap.set_over(over)

    def _annotate_heatmap(self, ax, mesh):
        """Add textual labels with the value in each cell."""
        mesh.update_scalarmappable()
        height, width = self.annot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        for x, y, m, color, val in zip(xpos.flat, ypos.flat, mesh.get_array().flat, mesh.get_facecolors(), self.annot_data.flat):
            if m is not np.ma.masked:
                lum = relative_luminance(color)
                text_color = '.15' if lum > 0.408 else 'w'
                annotation = ('{:' + self.fmt + '}').format(val)
                text_kwargs = dict(color=text_color, ha='center', va='center')
                text_kwargs.update(self.annot_kws)
                ax.text(x, y, annotation, **text_kwargs)

    def _skip_ticks(self, labels, tickevery):
        """Return ticks and labels at evenly spaced intervals."""
        n = len(labels)
        if tickevery == 0:
            ticks, labels = ([], [])
        elif tickevery == 1:
            ticks, labels = (np.arange(n) + 0.5, labels)
        else:
            start, end, step = (0, n, tickevery)
            ticks = np.arange(start, end, step) + 0.5
            labels = labels[start:end:step]
        return (ticks, labels)

    def _auto_ticks(self, ax, labels, axis):
        """Determine ticks and ticklabels that minimize overlap."""
        transform = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(transform)
        size = [bbox.width, bbox.height][axis]
        axis = [ax.xaxis, ax.yaxis][axis]
        tick, = axis.set_ticks([0])
        fontsize = tick.label1.get_size()
        max_ticks = int(size // (fontsize / 72))
        if max_ticks < 1:
            return ([], [])
        tick_every = len(labels) // max_ticks + 1
        tick_every = 1 if tick_every == 0 else tick_every
        ticks, labels = self._skip_ticks(labels, tick_every)
        return (ticks, labels)

    def plot(self, ax, cax, kws):
        """Draw the heatmap on the provided Axes."""
        despine(ax=ax, left=True, bottom=True)
        if kws.get('norm') is None:
            kws.setdefault('vmin', self.vmin)
            kws.setdefault('vmax', self.vmax)
        mesh = ax.pcolormesh(self.plot_data, cmap=self.cmap, **kws)
        ax.set(xlim=(0, self.data.shape[1]), ylim=(0, self.data.shape[0]))
        ax.invert_yaxis()
        if self.cbar:
            cb = ax.figure.colorbar(mesh, cax, ax, **self.cbar_kws)
            cb.outline.set_linewidth(0)
            if kws.get('rasterized', False):
                cb.solids.set_rasterized(True)
        if isinstance(self.xticks, str) and self.xticks == 'auto':
            xticks, xticklabels = self._auto_ticks(ax, self.xticklabels, 0)
        else:
            xticks, xticklabels = (self.xticks, self.xticklabels)
        if isinstance(self.yticks, str) and self.yticks == 'auto':
            yticks, yticklabels = self._auto_ticks(ax, self.yticklabels, 1)
        else:
            yticks, yticklabels = (self.yticks, self.yticklabels)
        ax.set(xticks=xticks, yticks=yticks)
        xtl = ax.set_xticklabels(xticklabels)
        ytl = ax.set_yticklabels(yticklabels, rotation='vertical')
        plt.setp(ytl, va='center')
        _draw_figure(ax.figure)
        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation='vertical')
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation='horizontal')
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        if self.annot:
            self._annotate_heatmap(ax, mesh)