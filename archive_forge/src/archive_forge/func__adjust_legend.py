import copy
import math
import warnings
from types import FunctionType
import matplotlib.colors as mpl_colors
import numpy as np
import param
from matplotlib import ticker
from matplotlib.dates import date2num
from matplotlib.image import AxesImage
from packaging.version import Version
from ...core import (
from ...core.options import Keywords, abbreviated_exception
from ...element import Graph, Path
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_range_key, process_cmap
from .plot import MPLPlot, mpl_rc_context
from .util import EqHistNormalize, mpl_version, validate, wrap_formatter
def _adjust_legend(self, overlay, axis):
    """
        Accumulate the legend handles and labels for all subplots
        and set up the legend
        """
    legend_data = []
    legend_plot = True
    dimensions = overlay.kdims
    title = ', '.join([d.label for d in dimensions])
    labels = self.legend_labels
    for key, subplot in self.subplots.items():
        element = overlay.data.get(key, False)
        if not subplot.show_legend or not element:
            continue
        title = ', '.join([d.name for d in dimensions])
        handle = subplot.traverse(lambda p: p.handles['artist'], [lambda p: 'artist' in p.handles])
        if getattr(subplot, '_legend_plot', None) is not None:
            legend_plot = True
        elif isinstance(overlay, NdOverlay):
            label = ','.join([dim.pprint_value(k, print_unit=True) for k, dim in zip(key, dimensions)])
            if handle:
                legend_data.append((handle, label))
        elif isinstance(subplot, OverlayPlot):
            legend_data += subplot.handles.get('legend_data', {}).items()
        elif element.label and handle:
            legend_data.append((handle, labels.get(element.label, element.label)))
    all_handles, all_labels = list(zip(*legend_data)) if legend_data else ([], [])
    data = {}
    used_labels = []
    for handle, label in zip(all_handles, all_labels):
        if isinstance(handle, list):
            handle = tuple(handle)
        handle = tuple((h for h in handle if not isinstance(h, (AxesImage, list))))
        if not handle:
            continue
        if handle and handle not in data and label and (label not in used_labels):
            data[handle] = label
            used_labels.append(label)
    if not len(set(data.values())) > 0 or not self.show_legend:
        legend = axis.get_legend()
        if legend and (not (legend_plot or self.show_legend)):
            legend.set_visible(False)
    else:
        leg = axis.legend(list(data.keys()), list(data.values()), title=title, **self._legend_opts)
        title_fontsize = self._fontsize('legend_title')
        if title_fontsize:
            leg.get_title().set_fontsize(title_fontsize['fontsize'])
        leg.set_zorder(10000000.0)
        self.handles['legend'] = leg
        self.handles['bbox_extra_artists'].append(leg)
    self.handles['legend_data'] = data