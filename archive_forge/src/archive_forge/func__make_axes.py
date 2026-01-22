from collections import defaultdict
from itertools import groupby
import numpy as np
import param
from bokeh.layouts import gridplot
from bokeh.models import (
from bokeh.models.layouts import TabPanel, Tabs
from ...core import (
from ...core.options import SkipRendering
from ...core.util import (
from ...selection import NoOpSelectionDisplay
from ..links import Link
from ..plot import (
from ..util import attach_streams, collate, displayable
from .links import LinkCallback
from .util import (
def _make_axes(self, plot):
    width, height = self.renderer.get_size(plot)
    x_axis, y_axis = (None, None)
    keys = self.layout.keys(full_grid=True)
    if self.xaxis:
        flip = self.shared_xaxis
        rotation = self.xrotation
        lsize = self._fontsize('xlabel').get('fontsize')
        tsize = self._fontsize('xticks', common=False).get('fontsize')
        xfactors = list(unique_iterator([wrap_tuple(k)[0] for k in keys]))
        x_axis = make_axis('x', width, xfactors, self.layout.kdims[0], flip=flip, rotation=rotation, label_size=lsize, tick_size=tsize)
    if self.yaxis and self.layout.ndims > 1:
        flip = self.shared_yaxis
        rotation = self.yrotation
        lsize = self._fontsize('ylabel').get('fontsize')
        tsize = self._fontsize('yticks', common=False).get('fontsize')
        yfactors = list(unique_iterator([k[1] for k in keys]))
        y_axis = make_axis('y', height, yfactors, self.layout.kdims[1], flip=flip, rotation=rotation, label_size=lsize, tick_size=tsize)
    if x_axis and y_axis:
        plot = filter_toolboxes(plot)
        r1, r2 = ([y_axis, plot], [None, x_axis])
        if self.shared_xaxis:
            r1, r2 = (r2, r1)
        if self.shared_yaxis:
            x_axis.margin = (0, 0, 0, 50)
            r1, r2 = (r1[::-1], r2[::-1])
        plot = gridplot([r1, r2], merge_tools=False)
        if self.merge_tools:
            plot.toolbar = merge_tools([r1, r2])
    elif y_axis:
        models = [y_axis, plot]
        if self.shared_yaxis:
            models = models[::-1]
        plot = Row(*models)
    elif x_axis:
        models = [plot, x_axis]
        if self.shared_xaxis:
            models = models[::-1]
        plot = Column(*models)
    return plot